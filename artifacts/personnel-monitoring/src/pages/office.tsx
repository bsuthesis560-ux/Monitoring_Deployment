import React, { useEffect, useRef, useState } from "react";
import { useLocation, useRoute } from "wouter";
import { AppLayout } from "@/components/layout/AppLayout";
import { format } from "date-fns";
import {
  Building2,
  Search,
  Camera,
  WifiOff,
  Clock,
  LogIn,
  LogOut as LogOutIcon,
  ChevronRight,
  ArrowRight,
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { motion } from "framer-motion";
import {
  getOfficeBySlug,
  useDepartment,
} from "@/contexts/department-context";

interface MonitoringLog {
  id: number;
  employeeId: string;
  name: string;
  department: string;
  logType: string;
  timestamp: string;
}

interface DeptStat { department: string; total: number; }

const POLL_INTERVAL = 5000;

export default function OfficePage() {
  const [, params] = useRoute<{ slug: string }>("/office/:slug");
  const [, setLocation] = useLocation();
  const { setSelectedDept } = useDepartment();

  const office = params?.slug ? getOfficeBySlug(params.slug) : undefined;

  const [searchTerm, setSearchTerm] = useState("");
  const [logs, setLogs] = useState<MonitoringLog[]>([]);
  const [stats, setStats] = useState<DeptStat[]>([]);
  const [logsError, setLogsError] = useState(false);
  const [activeUnit, setActiveUnit] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchLogs = async () => {
    try {
      const res = await fetch("/api/logs", { credentials: "include" });
      if (res.ok) {
        setLogs(await res.json());
        setLastUpdated(new Date());
        setLogsError(false);
      } else {
        setLogsError(true);
      }
    } catch {
      setLogsError(true);
    }
  };

  useEffect(() => {
    fetch("/api/stats", { credentials: "include" })
      .then(r => r.json())
      .then(d => setStats(Array.isArray(d) ? d : []))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!office) return;
    fetchLogs();
    intervalRef.current = setInterval(fetchLogs, POLL_INTERVAL);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [office?.slug]);

  // Reset active unit when office changes
  useEffect(() => { setActiveUnit(null); }, [office?.slug]);

  if (!office) {
    return (
      <AppLayout>
        <div className="h-[60vh] flex flex-col items-center justify-center text-center">
          <h2 className="text-2xl font-bold text-gray-800">Office not found</h2>
          <button
            onClick={() => setLocation("/dashboard")}
            className="mt-4 px-5 py-2 bg-primary text-white rounded-xl font-semibold hover:bg-primary/90"
          >
            Back to Dashboard
          </button>
        </div>
      </AppLayout>
    );
  }

  const officeUnits = new Set(office.units);

  const officeFilteredLogs = logs.filter(l => officeUnits.has(l.department));
  const finalLogs = (activeUnit
    ? officeFilteredLogs.filter(l => l.department === activeUnit)
    : officeFilteredLogs
  ).filter(l =>
    l.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    l.employeeId.includes(searchTerm)
  );

  const getUnitTotal = (unit: string) =>
    stats.find(s => s.department === unit)?.total ?? 0;

  const handleOpenUnit = (unit: string) => {
    setSelectedDept(unit);
    setLocation("/staff-monitoring");
  };

  return (
    <AppLayout>
      <div className="flex flex-col gap-4 h-[calc(100vh-10rem)]">

        {/* Header */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Building2 className="w-5 h-5 text-primary" />
            </div>
            <div>
              <p className="text-[10px] font-bold text-primary tracking-[0.25em] uppercase">{office.code}</p>
              <h2 className="text-lg sm:text-xl font-bold text-gray-900 leading-tight">{office.name}</h2>
              <div className="flex items-center gap-2 mt-1">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs text-gray-500">
                  Live · updates every 5s
                  {lastUpdated && ` · Last: ${format(lastUpdated, "hh:mm:ss aa")}`}
                </span>
              </div>
            </div>
          </div>
          <div className="relative w-full sm:w-72">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <Input
              placeholder="Search by name or ID..."
              className="pl-9 h-10 bg-white"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>

        {/* Body: Sub-units sidebar + Monitoring log */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-0">

          {/* Sub-units list */}
          <aside className="lg:col-span-4 xl:col-span-3 bg-white rounded-2xl shadow-sm border border-gray-200 flex flex-col min-h-0 overflow-hidden">
            <div className="p-4 border-b border-gray-100 flex-shrink-0">
              <p className="text-xs font-bold text-gray-500 uppercase tracking-wider">Sub-units</p>
              <p className="text-xs text-gray-400 mt-0.5">Click to filter the log on the right.</p>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
              <button
                onClick={() => setActiveUnit(null)}
                className={`w-full text-left px-4 py-3 rounded-xl border transition-all text-sm font-semibold ${
                  activeUnit === null
                    ? "bg-primary text-white border-primary shadow-sm"
                    : "bg-white text-gray-700 border-gray-200 hover:border-primary/40 hover:bg-primary/5"
                }`}
              >
                All Sub-units
                <span className={`ml-2 text-[11px] font-bold ${activeUnit === null ? "text-white/80" : "text-gray-400"}`}>
                  ({office.units.length})
                </span>
              </button>

              {office.units.map((unit, idx) => {
                const isActive = activeUnit === unit;
                const total = getUnitTotal(unit);
                return (
                  <motion.div
                    key={unit}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.25, delay: idx * 0.03 }}
                  >
                    <button
                      onClick={() => setActiveUnit(unit)}
                      className={`w-full text-left px-4 py-3 rounded-xl border transition-all text-sm group ${
                        isActive
                          ? "bg-primary/5 border-primary text-primary shadow-sm"
                          : "bg-white text-gray-700 border-gray-200 hover:border-primary/40 hover:bg-primary/5"
                      }`}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <p className={`font-semibold leading-tight ${isActive ? "text-primary" : "text-gray-800"}`}>
                            {unit}
                          </p>
                          <p className="text-[11px] text-gray-400 mt-1">
                            {total} {total === 1 ? "personnel" : "personnel"}
                          </p>
                        </div>
                        <ChevronRight className={`w-4 h-4 mt-0.5 transition-all ${isActive ? "text-primary translate-x-0.5" : "text-gray-300 group-hover:text-primary"}`} />
                      </div>
                      {isActive && (
                        <div className="mt-3 pt-3 border-t border-primary/20">
                          <div
                            role="button"
                            tabIndex={0}
                            onClick={(e) => { e.stopPropagation(); handleOpenUnit(unit); }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                e.stopPropagation();
                                handleOpenUnit(unit);
                              }
                            }}
                            className="text-xs font-semibold inline-flex items-center gap-1 text-primary hover:underline cursor-pointer"
                          >
                            Open full monitoring view
                            <ArrowRight className="w-3 h-3" />
                          </div>
                        </div>
                      )}
                    </button>
                  </motion.div>
                );
              })}
            </div>
          </aside>

          {/* Monitoring Log */}
          <section className="lg:col-span-8 xl:col-span-9 bg-white rounded-2xl shadow-sm border border-gray-200 flex flex-col min-h-0 overflow-hidden">
            <div className="p-4 border-b border-gray-100 flex items-center gap-2 flex-shrink-0">
              <Clock className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wider">
                Monitoring Log
                {activeUnit && (
                  <span className="ml-2 normal-case font-medium text-gray-500">
                    · filtered to <span className="text-primary font-semibold">{activeUnit}</span>
                  </span>
                )}
              </h3>
              <span className="ml-auto text-xs text-gray-400 font-medium">
                {finalLogs.length} {finalLogs.length === 1 ? "entry" : "entries"}
              </span>
            </div>

            <div className="flex-1 overflow-auto">
              {logsError ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-gray-400 gap-3">
                  <WifiOff className="w-10 h-10" />
                  <p className="font-medium">Could not load monitoring logs.</p>
                </div>
              ) : finalLogs.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-gray-400 gap-3">
                  <Camera className="w-10 h-10" />
                  <p className="font-medium text-gray-600">
                    No monitoring logs yet for {activeUnit ?? office.code}.
                  </p>
                  <p className="text-sm text-center max-w-sm">
                    Start the facial recognition service on your local machine to begin logging monitoring events.
                  </p>
                </div>
              ) : (
                <table className="w-full text-left border-collapse">
                  <thead className="bg-gray-50 sticky top-0 z-10 border-b border-gray-200">
                    <tr>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Name</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Department</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Date & Time</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Log Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {finalLogs.map((log) => {
                      const isTimeIn = log.logType === "TIME_IN";
                      const dt = new Date(log.timestamp);
                      return (
                        <tr key={log.id} className="hover:bg-blue-50/30 transition-colors">
                          <td className="py-3 px-5">
                            <div className="flex items-center gap-2">
                              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${isTimeIn ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
                                {log.name.charAt(0)}
                              </div>
                              <p className="font-semibold text-gray-900 text-sm">{log.name}</p>
                            </div>
                          </td>
                          <td className="py-3 px-5">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200">
                              {log.department}
                            </span>
                          </td>
                          <td className="py-3 px-5 text-sm text-gray-700 whitespace-nowrap">
                            {format(dt, "MM/dd/yyyy")}
                            <span className="ml-2 font-mono text-gray-500">{format(dt, "hh:mm:ss aa")}</span>
                          </td>
                          <td className="py-3 px-5">
                            {isTimeIn ? (
                              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-green-100 text-green-800 border border-green-200">
                                <LogIn className="w-3 h-3" />
                                TIME IN
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-red-100 text-red-800 border border-red-200">
                                <LogOutIcon className="w-3 h-3" />
                                TIME OUT
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
            </div>
          </section>
        </div>
      </div>
    </AppLayout>
  );
}
