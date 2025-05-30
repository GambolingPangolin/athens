import asyncio
import tempfile
import shutil
import os

import aiofiles

LEAN_PROJECT_PATH = "lean-project"


async def run_lake_lean_example(text: str):
    import aiofiles

    proc = None

    # Use the current folder
    async with aiofiles.tempfile.NamedTemporaryFile(
        mode="w", prefix="Example", suffix=".lean", dir=LEAN_PROJECT_PATH
    ) as tmpsrc:
        await tmpsrc.write(text)
        await tmpsrc.flush()

        filename = os.path.basename(tmpsrc.name)
        proc = await asyncio.create_subprocess_exec(
            "lake",
            "lean",
            filename,
            cwd=LEAN_PROJECT_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

    return proc.returncode, stdout.decode(), stderr.decode()
