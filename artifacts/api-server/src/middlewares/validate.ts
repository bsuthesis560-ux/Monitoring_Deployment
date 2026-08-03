import { z, ZodSchema } from "zod";
import type { Request, Response, NextFunction } from "express";

/**
 * Express middleware that validates req.body against a Zod schema.
 * On failure it returns 400 with field-level error details.
 * On success it replaces req.body with the parsed (trimmed / stripped) data.
 */
export function validate(schema: ZodSchema) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      res.status(400).json({
        error: "Validation failed",
        details: result.error.flatten().fieldErrors,
      });
      return;
    }
    req.body = result.data; // replace with parsed & stripped data
    next();
  };
}
