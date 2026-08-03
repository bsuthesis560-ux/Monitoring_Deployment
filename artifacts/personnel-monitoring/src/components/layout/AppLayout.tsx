import React from "react";
import { Link, useLocation } from "wouter";
import { useAuth } from "@/hooks/use-auth";
import { useDepartment, getOfficeForUnit } from "@/contexts/department-context";
import { Header } from "./Header";
import { LogOut, LayoutDashboard, Users, UserPlus, Shield, Home, ChevronRight, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";

const BSU_BG_URL = `${import.meta.env.BASE_URL}images/bsu-logo-bg.png`;

export function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const { selectedDept, clearDept } = useDepartment();
  const [location] = useLocation();

  if (!user) return null;

  const isAdmin = user.role === "admin";
  const selectedOffice = selectedDept ? getOfficeForUnit(selectedDept) : undefined;

  const navigation = [
    { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard, adminOnly: true },
    { name: "Staff Monitoring", href: "/staff-monitoring", icon: Users, adminOnly: false },
    { name: "Register Personnel", href: "/register", icon: UserPlus, adminOnly: true },
    { name: "Manage Accounts", href: "/accounts", icon: Shield, adminOnly: true },
  ];

  return (
    <div
      className="min-h-screen flex flex-col bg-gray-50/50 relative"
      style={{
        backgroundImage: `url(${BSU_BG_URL})`,
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center 55%",
        backgroundSize: "min(640px, 70vw)",
        backgroundAttachment: "fixed",
        backgroundColor: "rgba(249, 250, 251, 0.96)",
        backgroundBlendMode: "lighten",
      }}
    >
      <Header />

      <div className="flex-1 flex w-full max-w-[1600px] mx-auto overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-white border-r border-gray-200 flex-shrink-0 hidden lg:flex flex-col shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)] z-0">
          {/* User info */}
          <div className="p-5 border-b border-gray-100">
            <div className="flex items-center gap-3">
              <div className="w-11 h-11 rounded-full bg-primary text-white flex items-center justify-center font-bold text-lg shadow-md">
                {user.name ? user.name.charAt(0).toUpperCase() : user.username.charAt(0).toUpperCase()}
              </div>
              <div className="overflow-hidden">
                <p className="font-bold text-gray-900 truncate text-sm">{user.name || user.username}</p>
                <p className="text-xs font-semibold text-primary uppercase tracking-wider">{user.role}</p>
              </div>
            </div>
          </div>

          {/* Sub-unit context banner (admin only) */}
          {isAdmin && (
            <div className={`mx-3 mt-3 rounded-xl px-4 py-3 border ${selectedDept ? "bg-primary/5 border-primary/20" : "bg-amber-50 border-amber-200"}`}>
              {selectedDept ? (
                <div>
                  <p className="text-xs font-bold text-primary/60 uppercase tracking-wider">Active Sub-unit</p>
                  <p className="text-sm font-bold text-primary mt-0.5 leading-tight">{selectedDept}</p>
                  {selectedOffice && (
                    <p className="text-[10px] font-semibold text-gray-500 mt-1 uppercase tracking-wider">
                      {selectedOffice.code}
                    </p>
                  )}
                  <button
                    onClick={clearDept}
                    className="mt-2 flex items-center gap-1 text-xs text-gray-500 hover:text-primary transition-colors font-medium"
                  >
                    <RefreshCw className="w-3 h-3" />
                    Change sub-unit
                  </button>
                </div>
              ) : (
                <div>
                  <p className="text-xs font-bold text-amber-600 uppercase tracking-wider">No Sub-unit</p>
                  <p className="text-xs text-amber-700 mt-0.5">Pick an office from the Dashboard</p>
                </div>
              )}
            </div>
          )}

          <nav className="flex-1 p-3 space-y-1 overflow-y-auto mt-2">
            {navigation.map((item) => {
              if (item.adminOnly && !isAdmin) return null;
              const isActive = location === item.href || location.startsWith(item.href + "?");
              return (
                <Link key={item.name} href={item.href}>
                  <div className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-xl transition-all cursor-pointer font-medium text-sm",
                    isActive
                      ? "bg-primary text-white shadow-md shadow-primary/20"
                      : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
                  )}>
                    <item.icon className={cn("w-4 h-4", isActive ? "text-white" : "text-gray-500")} />
                    {item.name}
                  </div>
                </Link>
              );
            })}
          </nav>

          <div className="p-3 border-t border-gray-100">
            <button
              onClick={logout}
              className="flex items-center gap-3 px-4 py-3 w-full rounded-xl text-red-600 font-medium hover:bg-red-50 transition-colors text-sm"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
          {/* Top bar */}
          <div className="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6 lg:px-8 shadow-sm z-10 sticky top-0">
            <div className="flex items-center gap-2">
              {isAdmin && location !== "/dashboard" && (
                <Link href="/dashboard">
                  <button className="flex items-center gap-1 text-sm text-gray-500 hover:text-primary font-medium mr-2 transition-colors">
                    <Home className="w-4 h-4" />
                    <span className="hidden sm:inline">Dashboard</span>
                    <ChevronRight className="w-3 h-3" />
                  </button>
                </Link>
              )}
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-gray-800">
                  {navigation.find(n => n.href === location || location.startsWith(n.href + "?"))?.name ?? "Dashboard"}
                </h2>
                {isAdmin && selectedDept && location === "/staff-monitoring" && (
                  <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
                    {selectedDept}
                  </span>
                )}
              </div>
            </div>

            <div className="lg:hidden">
              <button onClick={logout} className="p-2 text-gray-500 hover:text-red-600 hover:bg-red-50 rounded-full">
                <LogOut className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-auto p-4 md:p-6 lg:p-8 relative z-0">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
