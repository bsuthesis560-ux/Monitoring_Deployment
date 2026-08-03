import React, { useState, useRef, useEffect } from "react";
import { useLocation } from "wouter";
import { AppLayout } from "@/components/layout/AppLayout";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { UploadCloud, CheckCircle2, AlertCircle, Info, X, Pencil } from "lucide-react";
import { motion } from "framer-motion";
import { OFFICES } from "@/contexts/department-context";

// ── Validation helpers ────────────────────────────────────────────────────────
// Allows: letters (incl. accented), spaces, apostrophes, hyphens, periods
// Covers suffixes like Jr., Sr., II, III, IV. Numbers and symbols are rejected.
const NAME_PATTERN = /^[a-zA-ZÀ-ÿ\s.''\-]+$/;
const NAME_MSG     = "Only letters, spaces, hyphens, and apostrophes are allowed.";
const NAME_MAX     = 100;

function nameField(requiredMsg: string) {
  return z
    .string()
    .trim()
    .min(1, requiredMsg)
    .max(NAME_MAX, `Must be ${NAME_MAX} characters or fewer`)
    .regex(NAME_PATTERN, NAME_MSG);
}

const registerSchema = z.object({
  lastName:   nameField("Last name is required"),
  firstName:  nameField("First name is required"),
  middleName: z.union([
    z.string().trim().max(NAME_MAX, `Must be ${NAME_MAX} characters or fewer`).regex(NAME_PATTERN, NAME_MSG),
    z.literal(""),
  ]).optional(),
  employeeId: z
    .string()
    .min(10, "Employee ID must be exactly 10 characters")
    .max(10, "Employee ID must be exactly 10 characters")
    .regex(/^[0-9\-]+$/, "Numbers and hyphens (-) only"),
  department:    z.string().min(1, "Department is required"),
  position:      z.string().min(1, "Position is required"),
  createAccount: z.boolean().default(false),
  password:      z.string().optional(),
}).refine(data => {
  if (data.createAccount && (!data.password || data.password.length < 6)) return false;
  return true;
}, { message: "Password is required and must be at least 6 characters", path: ["password"] });

type RegisterFormValues = z.infer<typeof registerSchema>;

// ── Name normalization ────────────────────────────────────────────────────────
const ROMAN_NUMERALS = new Set(["II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII"]);
const LOWER_SUFFIXES  = new Set(["jr.", "sr.", "md.", "phd.", "dds."]);

function toTitleCase(str: string): string {
  return str
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .map(word => {
      if (!word) return word;
      if (ROMAN_NUMERALS.has(word.toUpperCase())) return word.toUpperCase();
      if (LOWER_SUFFIXES.has(word.toLowerCase())) {
        return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
      }
      return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
    })
    .join(" ");
}

// ── Photo upload component ────────────────────────────────────────────────────
type ViewType = "front" | "left" | "right" | "top";

const VIEW_LABELS: Record<ViewType, { label: string; hint: string; required: boolean }> = {
  front: { label: "Front View",    hint: "Face forward, eyes level",          required: true  },
  left:  { label: "Left Profile",  hint: "Turn head to the left",             required: false },
  right: { label: "Right Profile", hint: "Turn head to the right",            required: false },
  top:   { label: "Top View",      hint: "Camera slightly above eye level",   required: false },
};

const VIEW_ORDER: ViewType[] = ["front", "left", "right", "top"];

interface PhotoSlotProps {
  viewType: ViewType;
  preview: string | null;
  hasError: boolean;
  onFileChange: (viewType: ViewType, file: File) => void;
  onRemove: (viewType: ViewType) => void;
}

function PhotoSlot({ viewType, preview, hasError, onFileChange, onRemove }: PhotoSlotProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const meta = VIEW_LABELS[viewType];

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 3 * 1024 * 1024) { alert("Photo must be under 3MB."); return; }
    onFileChange(viewType, file);
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold text-gray-700">
          {meta.label}
          {meta.required && <span className="text-red-500 ml-0.5">*</span>}
        </p>
        {hasError && (
          <span className="flex items-center gap-0.5 text-xs text-red-500">
            <AlertCircle className="w-3 h-3" /> Required
          </span>
        )}
      </div>
      <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={handleChange} />
      <div
        onClick={() => inputRef.current?.click()}
        className={`relative rounded-xl border-dashed border-2 flex flex-col items-center justify-center h-36 cursor-pointer overflow-hidden transition-colors group
          ${hasError ? "border-red-400 bg-red-50/30" : preview ? "border-primary/40 bg-primary/5" : "border-gray-200 hover:border-primary/40 hover:bg-gray-50"}`}
      >
        {preview ? (
          <>
            <img src={preview} alt={meta.label} className="w-full h-full object-cover" />
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onRemove(viewType); }}
              className="absolute top-1.5 right-1.5 bg-white/90 hover:bg-white rounded-full p-0.5 shadow"
            >
              <X className="w-3.5 h-3.5 text-gray-600" />
            </button>
          </>
        ) : (
          <>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center mb-1.5 transition-transform group-hover:scale-110 ${hasError ? "bg-red-100" : "bg-gray-100"}`}>
              <UploadCloud className={`w-5 h-5 ${hasError ? "text-red-400" : "text-gray-400 group-hover:text-primary"}`} />
            </div>
            <p className="text-xs font-medium text-gray-600">Click to upload</p>
            <p className="text-xs text-gray-400 mt-0.5">{meta.hint}</p>
          </>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
interface RegisterProps { id?: string; }

export default function Register({ id }: RegisterProps) {
  const [, setLocation] = useLocation();
  const isUpdateMode = !!id;
  const personnelId  = id ? parseInt(id) : null;

  const [success,     setSuccess]     = useState(false);
  const [photos,      setPhotos]      = useState<Partial<Record<ViewType, string>>>({});
  const [photoError,  setPhotoError]  = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [loading,     setLoading]     = useState(isUpdateMode);
  const [isPending,   setIsPending]   = useState(false);

  const { register, handleSubmit, watch, reset, setValue, formState: { errors } } = useForm<RegisterFormValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: { createAccount: false },
  });

  const watchCreateAccount = watch("createAccount");

  // Load existing record in update mode
  useEffect(() => {
    if (!isUpdateMode || !personnelId) return;
    setLoading(true);
    fetch(`/api/personnel/${personnelId}`, { credentials: "include" })
      .then(r => r.json())
      .then(data => {
        reset({
          lastName:      data.lastName    ?? "",
          firstName:     data.firstName   ?? "",
          middleName:    data.middleInitial ?? "",
          employeeId:    data.employeeId  ?? "",
          department:    data.department  ?? "",
          position:      data.position    ?? "",
          createAccount: false,
        });
        if (data.photos) {
          const loaded: Partial<Record<ViewType, string>> = {};
          for (const vt of VIEW_ORDER) { if (data.photos[vt]) loaded[vt] = data.photos[vt]; }
          setPhotos(loaded);
        } else if (data.photoUrl) {
          setPhotos({ front: data.photoUrl });
        }
      })
      .catch(() => setServerError("Failed to load personnel data."))
      .finally(() => setLoading(false));
  }, [personnelId]);

  const handleFileChange = (viewType: ViewType, file: File) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      setPhotos(prev => ({ ...prev, [viewType]: ev.target?.result as string }));
      if (viewType === "front") setPhotoError(false);
    };
    reader.readAsDataURL(file);
  };

  const handleRemove = (viewType: ViewType) => {
    setPhotos(prev => { const next = { ...prev }; delete next[viewType]; return next; });
  };

  // Block invalid characters in Employee ID field
  const handleEmployeeIdKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    const allowed = new Set(["Backspace","Delete","Tab","ArrowLeft","ArrowRight","Home","End","-"]);
    if (allowed.has(e.key)) return;
    if (/^\d$/.test(e.key)) return;
    e.preventDefault();
  };

  const onSubmit = async (data: RegisterFormValues) => {
    if (!photos.front) { setPhotoError(true); return; }
    setServerError(null);
    setPhotoError(false);
    setIsPending(true);

    // Normalize names to Title Case before saving
    const normalizedLastName   = toTitleCase(data.lastName);
    const normalizedFirstName  = toTitleCase(data.firstName);
    const normalizedMiddleName = data.middleName ? toTitleCase(data.middleName) : undefined;

    try {
      if (isUpdateMode && personnelId) {
        const res = await fetch(`/api/personnel/${personnelId}`, {
          method: "PUT",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lastName:      normalizedLastName,
            firstName:     normalizedFirstName,
            middleInitial: normalizedMiddleName,
            department:    data.department,
            position:      data.position,
            photoUrl: photos.front,
            photos: {
              front: photos.front  ?? null,
              left:  photos.left   ?? null,
              right: photos.right  ?? null,
              top:   photos.top    ?? null,
            },
          }),
        });
        if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Failed to update."); }
        setSuccess(true);
        setTimeout(() => setLocation("/staff-monitoring"), 2000);
      } else {
        const res = await fetch("/api/personnel", {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            lastName:      normalizedLastName,
            firstName:     normalizedFirstName,
            middleInitial: normalizedMiddleName,
            employeeId:    data.employeeId,
            department:    data.department,
            position:      data.position,
            photoUrl: photos.front,
            photos: {
              front: photos.front  ?? null,
              left:  photos.left   ?? null,
              right: photos.right  ?? null,
              top:   photos.top    ?? null,
            },
            createAccount: data.createAccount,
            password: data.createAccount ? data.password : undefined,
          }),
        });
        if (!res.ok) { const err = await res.json(); throw new Error(err.error || "Failed to create."); }
        setSuccess(true);
        setTimeout(() => setLocation("/dashboard"), 2000);
      }
    } catch (err: any) {
      setServerError(err.message || "An unexpected error occurred.");
    } finally {
      setIsPending(false);
    }
  };

  const uploadedCount = VIEW_ORDER.filter(v => photos[v]).length;

  if (loading) {
    return (
      <AppLayout>
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
        </div>
      </AppLayout>
    );
  }

  return (
    <AppLayout>
      <div className="max-w-5xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-1">
            {isUpdateMode && (
              <div className="w-9 h-9 rounded-xl bg-amber-100 flex items-center justify-center">
                <Pencil className="w-5 h-5 text-amber-600" />
              </div>
            )}
            <h2 className="text-3xl font-bold text-gray-900">
              {isUpdateMode ? "Update Personnel" : "Register Personnel"}
            </h2>
          </div>
          <p className="text-gray-500 mt-1">
            {isUpdateMode
              ? "Modify existing personnel information. Employee ID cannot be changed."
              : "Add a new staff member to the monitoring system."}
          </p>
        </div>

        {success ? (
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-green-50 border-2 border-green-200 rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-lg"
          >
            <CheckCircle2 className="w-16 h-16 text-green-500 mb-4" />
            <h3 className="text-2xl font-bold text-green-800">
              {isUpdateMode ? "Update Successful!" : "Registration Successful!"}
            </h3>
            <p className="text-green-600 mt-2">
              {isUpdateMode
                ? "The personnel record has been updated. Redirecting..."
                : "The personnel record has been created. Redirecting to dashboard..."}
            </p>
          </motion.div>
        ) : (
          <form onSubmit={handleSubmit(onSubmit)} className="grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* Left Column — Multi-angle Photo Upload */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-gray-700">Facial Photos</p>
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${uploadedCount === 4 ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"}`}>
                  {uploadedCount}/4 uploaded
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {VIEW_ORDER.map((vt) => (
                  <PhotoSlot
                    key={vt}
                    viewType={vt}
                    preview={photos[vt] ?? null}
                    hasError={vt === "front" && photoError}
                    onFileChange={handleFileChange}
                    onRemove={handleRemove}
                  />
                ))}
              </div>

              <div className="bg-amber-50 border border-amber-200 rounded-xl p-3">
                <div className="flex items-start gap-2">
                  <Info className="w-4 h-4 text-amber-600 mt-0.5 flex-shrink-0" />
                  <div className="text-xs text-amber-700 space-y-1">
                    <p className="font-semibold">Photo Requirements</p>
                    <ul className="space-y-0.5 text-amber-600">
                      <li>• Front view is required</li>
                      <li>• All 4 angles improve accuracy</li>
                      <li>• Single person, good lighting</li>
                      <li>• No sunglasses or face covering</li>
                      <li>• Max 3MB per photo</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column — Form Fields */}
            <div className="lg:col-span-7 bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
              {serverError && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-xl text-sm border border-red-200 font-medium">
                  {serverError}
                </div>
              )}

              <div className="space-y-8">

                {/* Personnel Name */}
                <div>
                  <h3 className="text-lg font-bold text-gray-800 border-b border-gray-100 pb-2 mb-4">Personnel Name</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Last Name *</label>
                      <Input placeholder="e.g. Dela Cruz" {...register("lastName")} />
                      {errors.lastName && <p className="text-red-500 text-xs mt-1">{errors.lastName.message}</p>}
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">First Name *</label>
                      <Input placeholder="e.g. Juan" {...register("firstName")} />
                      {errors.firstName && <p className="text-red-500 text-xs mt-1">{errors.firstName.message}</p>}
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Middle Name</label>
                      <Input placeholder="e.g. Santos" {...register("middleName")} />
                      {errors.middleName && <p className="text-red-500 text-xs mt-1">{errors.middleName.message}</p>}
                    </div>
                  </div>
                  <p className="text-xs text-gray-400 mt-2">
                    Names will be automatically formatted to title case. Suffixes (Jr., Sr., II, III, IV) are accepted.
                  </p>
                </div>

                {/* Information */}
                <div>
                  <h3 className="text-lg font-bold text-gray-800 border-b border-gray-100 pb-2 mb-4">Information</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Employee ID *</label>
                      <Input
                        placeholder="e.g. 12345"
                        maxLength={10}
                        {...register("employeeId")}
                        onKeyDown={handleEmployeeIdKeyDown}
                        disabled={isUpdateMode}
                        className={isUpdateMode ? "bg-gray-100 text-gray-500 cursor-not-allowed" : ""}
                      />
                      <p className="text-xs text-gray-400 mt-1">Numbers and hyphens only · max 10 characters</p>
                      {errors.employeeId && <p className="text-red-500 text-xs mt-1">{errors.employeeId.message}</p>}
                    </div>
                    <div>
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Position *</label>
                      <select
                        {...register("position")}
                        className="w-full h-10 px-3 rounded-md border border-input bg-background text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      >
                        <option value="">Select Position</option>
                        <option value="Non-teaching">Non-teaching</option>
                        <optgroup label="Teaching">
                          <option value="Teaching - Permanent">Permanent</option>
                          <option value="Teaching - Temporary">Temporary</option>
                          <option value="Teaching - Guest Lecturer">Guest Lecturer</option>
                        </optgroup>
                      </select>
                      {errors.position && <p className="text-red-500 text-xs mt-1">{errors.position.message}</p>}
                    </div>
                    <div className="md:col-span-2">
                      <label className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1 block">Department / Sub-unit *</label>
                      <select
                        {...register("department")}
                        className="w-full h-10 px-3 rounded-md border border-input bg-background text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      >
                        <option value="">Select Department / Sub-unit</option>
                        {OFFICES.map(office => (
                          <optgroup key={office.slug} label={`${office.code} — ${office.name}`}>
                            {office.units.map(u => (
                              <option key={u} value={u}>{u}</option>
                            ))}
                          </optgroup>
                        ))}
                      </select>
                      {errors.department && <p className="text-red-500 text-xs mt-1">{errors.department.message}</p>}
                    </div>
                  </div>
                </div>

                {/* Account Settings — register mode only */}
                {!isUpdateMode && (
                  <div className="bg-gray-50 p-6 rounded-xl border border-gray-200">
                    <label className="flex items-start gap-3 cursor-pointer group">
                      <div className="mt-1 relative flex items-center justify-center">
                        <input type="checkbox" className="peer sr-only" {...register("createAccount")} />
                        <div className="w-5 h-5 border-2 border-gray-300 rounded bg-white peer-checked:bg-primary peer-checked:border-primary transition-colors"></div>
                        <svg className="absolute w-3.5 h-3.5 text-white pointer-events-none opacity-0 peer-checked:opacity-100" viewBox="0 0 14 10" fill="none">
                          <path d="M1 5L5 9L13 1" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <div>
                        <p className="font-semibold text-gray-900 group-hover:text-primary transition-colors">Create System Account</p>
                        <p className="text-sm text-gray-500 mt-0.5">Allows this person to log in and view their department's monitoring data.</p>
                      </div>
                    </label>
                    {watchCreateAccount && (
                      <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} className="mt-4 pl-8">
                        <Input type="password" placeholder="Assign a secure password (min. 6 characters)" {...register("password")} />
                        {errors.password && <p className="text-red-500 text-xs mt-1">{errors.password.message}</p>}
                      </motion.div>
                    )}
                  </div>
                )}

                {/* Actions */}
                <div className="pt-4 border-t border-gray-100 flex justify-end gap-3">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => setLocation(isUpdateMode ? "/staff-monitoring" : "/dashboard")}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    size="lg"
                    className={isUpdateMode
                      ? "bg-amber-600 hover:bg-amber-700 shadow-amber-600/20 text-white"
                      : "bg-green-600 hover:bg-green-700 shadow-green-600/20 text-white"}
                    disabled={isPending}
                  >
                    {isPending
                      ? (isUpdateMode ? "Updating..." : "Registering...")
                      : (isUpdateMode ? "Update Personnel" : "Register Personnel")}
                  </Button>
                </div>

              </div>
            </div>
          </form>
        )}
      </div>
    </AppLayout>
  );
}
