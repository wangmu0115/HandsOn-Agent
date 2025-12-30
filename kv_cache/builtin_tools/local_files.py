from dataclasses import dataclass
from pathlib import Path
from re import search
from typing import Any, Optional


@dataclass
class FileContentReadedResult:
    success: bool
    path: str
    content: str = ""
    total_lines: int = 0
    read_lines: int = 0
    offset: int = 0
    end: int = 0
    truncated: bool = False
    message: Optional[str] = None

    @classmethod
    def assuccess(
        cls,
        path: str,
    ):
        result = cls(True, path)
        return result

    @classmethod
    def exceed_file_length(cls, path: str, total_lines, offset):
        result = cls(True, path, content="", total_lines=total_lines, read_lines=0, offset=offset, end=0)
        result.message = f"Offset {offset} exceeds file length ({total_lines} lines)"
        return result

    @classmethod
    def file_not_found(cls, path: str):
        return FileContentReadedResult(False, path, message=f"File `{path}` not found.")

    @classmethod
    def error(cls, path: str, e: Exception):
        return FileContentReadedResult(False, path, message=str(e))


class LocalFileTools:
    """Local implementations of file system tools"""

    def __init__(self, root_dir: str = "/tmp"):
        self.root_dir = Path(root_dir)

    def read_file(
        self,
        file_path: str,
        offset: int = 0,
        size: Optional[int] = None,
    ) -> FileContentReadedResult:
        """
        Read contents of a file.

        Args:
            file_path: Path to the file relative to root directory
            offset: Line number to start reading from (0-based, default: 0)
            size: Number of lines to read (default: None, read all)

        Returns:
            Dictionary with file contents or error
        """
        fullpath = self.root_dir / file_path
        with open(fullpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        total_lines = len(lines)
        print(lines)

        # Apply offset and size
        read_offset = 0 if offset == 0 else offset
        if read_offset >= total_lines:
            return FileContentReadedResult.exceed_file_length(str(fullpath), total_lines, offset)
        # Determine end line
        if size is None:
            read_end = total_lines
        else:
            read_end = min(read_offset + size, total_lines)

    def find(self, pattern: str = "*", directory: str = ".") -> dict[str, Any]:
        """
        Find files matching a pattern (similar to Unix find command)

        Args:
            pattern: File name pattern (supports wildcards, default: "*" for all files)
            directory: Directory to search in (relative to root_dir)

        Returns:
            Dictionary with list of matching files
        """
        if directory == ".":
            search_dir = self.root_dir
        else:
            search_dir = self.root_dir / directory.strip("/")

        search_dir.walk()

    def grep(self, pattern: str, file_path: str = None, directory: str = None) -> dict[str, Any]:
        """
        Search for pattern in files (similar to Unix grep command)

        Args:
            pattern: Regular expression pattern to search for
            file_path: Single file to search in (optional)
            directory: Directory to search in (optional)

        Returns:
            Dictionary with matching lines
        """
