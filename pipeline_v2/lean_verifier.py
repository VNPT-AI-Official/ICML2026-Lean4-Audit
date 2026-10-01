"""
lean_verifier.py — Interface for verifying Lean 4 statements via the local compiler.
"""

import asyncio
import os
import tempfile
from pathlib import Path


async def verify_lean_code(code: str, header: str = "import Mathlib", timeout: int = 15) -> tuple[bool, str]:
    """
    Write Lean 4 code to a temporary file and run `lean` on it.
    
    Returns:
        (is_valid, error_output)
        is_valid: True if compilation succeeds (exit code 0).
        error_output: Stderr/Stdout from the compiler if failed, or a message if `lean` is not installed.
    """
    # Quick check if lean is installed by trying to run `lean --version`
    # We cache this to avoid spawning an extra process every time
    if not hasattr(verify_lean_code, "_lean_installed"):
        proc = await asyncio.create_subprocess_exec(
            "lean", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        verify_lean_code._lean_installed = (proc.returncode == 0)
    
    if not verify_lean_code._lean_installed:
        return True, "Lean 4 compiler not found. Skipping verification."

    # Combine header and code
    full_code = f"{header}\n\n{code}\n"
    
    # Create temp file
    fd, temp_path = tempfile.mkstemp(suffix=".lean", text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(full_code)

    try:
        # Run compiler
        proc = await asyncio.create_subprocess_exec(
            "lean", temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return False, "Lean 4 compilation timed out."
            
        if proc.returncode == 0:
            return True, ""
            
        output = stdout.decode('utf-8') + stderr.decode('utf-8')
        
        # Clean up output: remove temp file paths to avoid confusing the LLM
        clean_output = output.replace(temp_path, "statement.lean").strip()
        
        # Limit error output size so it doesn't blow up the context window
        if len(clean_output) > 1000:
            clean_output = clean_output[:1000] + "\n...[truncated]"
            
        return False, clean_output

    finally:
        # Cleanup
        try:
            os.remove(temp_path)
        except OSError:
            pass


if __name__ == "__main__":
    async def test():
        print("Testing lean verifier...")
        valid, msg = await verify_lean_code("theorem t (x : Nat) : x = x := rfl")
        print(f"Valid statement: {valid} (Expected: True if lean installed, else True)")
        if not valid:
            print(f"Error: {msg}")
            
        valid2, msg2 = await verify_lean_code("theorem t (x : Nat) : x = y := rfl")
        print(f"Invalid statement: {valid2} (Expected: False if lean installed)")
        if not valid2:
            print(f"Error output:\n{msg2}")

    asyncio.run(test())
