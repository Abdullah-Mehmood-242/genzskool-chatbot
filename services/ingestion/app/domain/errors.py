"""Domain errors for ingestion. They map to the shared error schema via common.errors."""

from __future__ import annotations

from common.errors import ProviderError, ValidationFailed


class DocumentError(ValidationFailed):
    """A source document cannot be ingested."""

    code = "document_error"


class CorruptDocument(DocumentError):
    code = "corrupt_document"
    default_message = "The PDF could not be parsed (corrupt, encrypted or not a PDF)."


class EmptyDocument(DocumentError):
    code = "empty_document"
    default_message = "The PDF has no pages or no extractable text."


class DocumentTooLarge(DocumentError):
    code = "document_too_large"
    default_message = "The PDF exceeds the configured size limit."


class InvalidFileName(DocumentError):
    code = "invalid_file_name"
    default_message = "File names must be plain PDF names inside the PDF directory."


class IndexWriteFailed(ProviderError):
    code = "index_write_failed"
    default_message = "The vector store did not persist all chunks."
