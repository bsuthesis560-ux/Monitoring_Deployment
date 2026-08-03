import * as React from "react"
import { cn } from "@/lib/utils"

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
  asChild?: boolean
}

export function buttonVariants({
  variant = "default",
  size = "default",
  className = "",
}: {
  variant?: ButtonProps["variant"]
  size?: ButtonProps["size"]
  className?: string
} = {}) {
  return cn(
    "inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 active:scale-[0.98]",
    {
      "bg-primary text-primary-foreground hover:bg-primary/90 shadow-md shadow-primary/20": variant === "default",
      "bg-destructive text-destructive-foreground hover:bg-destructive/90 shadow-md shadow-destructive/20": variant === "destructive",
      "border-2 border-input bg-background hover:bg-accent hover:text-accent-foreground": variant === "outline",
      "bg-secondary text-secondary-foreground hover:bg-secondary/80": variant === "secondary",
      "hover:bg-accent hover:text-accent-foreground": variant === "ghost",
      "text-primary underline-offset-4 hover:underline": variant === "link",
      "h-10 px-6 py-2": size === "default",
      "h-9 rounded-lg px-4": size === "sm",
      "h-12 rounded-xl px-8 text-base": size === "lg",
      "h-10 w-10": size === "icon",
    },
    className
  )
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    return (
      <button
        className={buttonVariants({ variant, size, className })}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button }
