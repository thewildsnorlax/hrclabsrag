"""Ingestion errors carry the HTTP status the API should return for them."""


class IngestionError(Exception):
    status_code = 400

    def __init__(self, message: str, filename: str = ""):
        super().__init__(message)
        self.message = message
        self.filename = filename


class UnsupportedFileType(IngestionError):
    status_code = 415


class FileTooLarge(IngestionError):
    status_code = 413


class TooManyPages(IngestionError):
    status_code = 413


class TooManyFiles(IngestionError):
    status_code = 400


class SessionDocumentLimit(IngestionError):
    status_code = 409


class UnreadableDocument(IngestionError):
    """Corrupt, encrypted, or has no extractable text (e.g. a scanned PDF)."""

    status_code = 422
