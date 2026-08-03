/**
 * P5 / P6 — Photo validation utility.
 *
 * Accepts only genuine JPEG and PNG data URIs up to MAX_PHOTO_BYTES.
 * This prevents SVG/GIF XSS vectors and enforces a per-photo size cap
 * independent of the Express body-size limit.
 */

const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png"];
const MAX_PHOTO_BYTES = 2 * 1024 * 1024; // 2 MB per photo

export interface PhotoValidationResult {
  valid: boolean;
  error?: string;
}

export function validatePhoto(dataUri: string): PhotoValidationResult {
  if (!dataUri.startsWith("data:")) {
    return { valid: false, error: "Invalid photo format" };
  }

  const mimeMatch = dataUri.match(/^data:([^;]+);base64,/);
  if (!mimeMatch) {
    return { valid: false, error: "Malformed data URI" };
  }

  const mimeType = mimeMatch[1].toLowerCase();
  if (!ALLOWED_MIME_TYPES.includes(mimeType)) {
    return {
      valid: false,
      error: `File type '${mimeType}' is not allowed. Only JPEG and PNG are accepted.`,
    };
  }

  const base64Data = dataUri.split(",")[1];
  if (!base64Data) {
    return { valid: false, error: "Empty photo data" };
  }

  // Approximate decoded byte size without actually decoding the buffer
  const byteSize = Math.ceil((base64Data.length * 3) / 4);
  if (byteSize > MAX_PHOTO_BYTES) {
    return { valid: false, error: "Photo exceeds maximum size of 2 MB" };
  }

  return { valid: true };
}

/**
 * Convenience: returns true if the data URI is a valid JPEG or PNG.
 * Mirrors the old `startsWith("data:image")` guard but is strictly safer.
 */
export function isValidPhoto(dataUri: string): boolean {
  return validatePhoto(dataUri).valid;
}
