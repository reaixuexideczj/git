import tempfile
import unittest
from pathlib import Path

from file_organizer import category_for, organize, unique_destination


class FileOrganizerTests(unittest.TestCase):
    def test_category_is_case_insensitive(self):
        self.assertEqual(category_for(Path("photo.JPG")), "images")
        self.assertEqual(category_for(Path("notes.md")), "documents")
        self.assertEqual(category_for(Path("unknown.xyz")), "others")

    def test_preview_does_not_move_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source = folder / "report.pdf"
            source.write_text("demo", encoding="utf-8")

            processed, _ = organize(folder, execute=False)

            self.assertEqual(processed, 1)
            self.assertTrue(source.exists())
            self.assertFalse((folder / "documents").exists())

    def test_execute_moves_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "photo.png").write_bytes(b"image")
            (folder / "program.py").write_text("print('ok')", encoding="utf-8")

            processed, _ = organize(folder, execute=True)

            self.assertEqual(processed, 2)
            self.assertTrue((folder / "images" / "photo.png").exists())
            self.assertTrue((folder / "code" / "program.py").exists())

    def test_existing_name_gets_number_suffix(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            existing = folder / "notes.txt"
            existing.write_text("old", encoding="utf-8")

            result = unique_destination(existing)

            self.assertEqual(result.name, "notes_1.txt")


if __name__ == "__main__":
    unittest.main()
