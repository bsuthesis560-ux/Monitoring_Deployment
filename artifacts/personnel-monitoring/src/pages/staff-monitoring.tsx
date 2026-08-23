import React, { useState, useEffect, useRef } from "react";
import { AppLayout } from "@/components/layout/AppLayout";
import { useAuth } from "@/hooks/use-auth";
import { useDepartment } from "@/contexts/department-context";
import { format, startOfWeek, endOfWeek, startOfMonth, endOfMonth, getWeek } from "date-fns";
import {
  Search, Users, Clock, WifiOff, Trash2, LayoutDashboard,
  LogIn, LogOut as LogOutIcon, BarChart3, Pencil, ChevronDown, ChevronRight, FileText,
  Download
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { useLocation } from "wouter";
import { exportDeptPDF, exportAllDeptsPDF, type SummaryPeriod } from "@/lib/pdf-export";

interface Personnel {
  id: number;
  lastName: string;
  firstName: string;
  middleInitial: string | null;
  employeeId: string;
  department: string;
  position: string;
  photoUrl: string | null;
  hasAccount: boolean;
  createdAt: string;
}

interface MonitoringLog {
  id: number;
  employeeId: string;
  name: string;
  department: string;
  position: string;
  logType: string;
  timestamp: string;
}

const POLL_INTERVAL = 5000;

export default function StaffMonitoring() {
  const { user } = useAuth();
  const { selectedDept } = useDepartment();
  const [, setLocation] = useLocation();
  const isAdmin = user?.role === "admin";

  const [searchTerm, setSearchTerm] = useState("");
  const [personnel, setPersonnel] = useState<Personnel[]>([]);
  const [logs, setLogs] = useState<MonitoringLog[]>([]);
  const [logsError, setLogsError] = useState(false);
  const [personnelLoading, setPersonnelLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [activeTab, setActiveTab] = useState<"logs" | "roster" | "summary">("logs");
  const [deleteConfirm, setDeleteConfirm] = useState<number | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Record Summary state
  const [summaryPeriod, setSummaryPeriod] = useState<SummaryPeriod>("daily");
  const [summaryDate, setSummaryDate]     = useState(() => format(new Date(), "yyyy-MM-dd"));
  const [customStartDate, setCustomStartDate] = useState(() => format(new Date(), "yyyy-MM-dd"));
  const [customEndDate,   setCustomEndDate]   = useState(() => format(new Date(), "yyyy-MM-dd"));
  const [customTimeFrom, setCustomTimeFrom]   = useState("00:00");
  const [customTimeTo,   setCustomTimeTo]     = useState("23:59");
  const [expandedDepts, setExpandedDepts] = useState<Set<string>>(new Set());
  const [exportingDept, setExportingDept] = useState<string | null>(null);
  const [exportingAll, setExportingAll] = useState(false);

  const deptForFetch = isAdmin ? selectedDept : null;

  const fetchPersonnel = async () => {
    setPersonnelLoading(true);
    try {
      const url = deptForFetch
        ? `/api/personnel?department=${encodeURIComponent(deptForFetch)}`
        : "/api/personnel";
      const res = await fetch(url, { credentials: "include" });
      if (res.ok) setPersonnel(await res.json());
    } finally {
      setPersonnelLoading(false);
    }
  };

  const fetchLogs = async () => {
    try {
      const base = deptForFetch
        ? `/api/logs?department=${encodeURIComponent(deptForFetch)}&limit=1000`
        : "/api/logs?limit=1000";
      const res = await fetch(base, { credentials: "include" });
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
    if (isAdmin && !selectedDept) return;
    fetchPersonnel();
    fetchLogs();
    intervalRef.current = setInterval(fetchLogs, POLL_INTERVAL);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [selectedDept]);

  const handleDelete = async (id: number) => {
    const res = await fetch(`/api/personnel/${id}`, { method: "DELETE", credentials: "include" });
    if (res.ok) {
      setPersonnel(prev => prev.filter(p => p.id !== id));
      setDeleteConfirm(null);
    }
  };

  const filteredLogs = logs.filter(l =>
    l.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    l.employeeId.includes(searchTerm)
  );

  const filteredPersonnel = personnel.filter(p =>
    `${p.firstName} ${p.lastName}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
    p.employeeId.includes(searchTerm)
  );

  // ── Record Summary helpers ──────────────────────────────────────────────────

  function getLogsForPeriod(): MonitoringLog[] {
    const selected = new Date(summaryDate + "T00:00:00");

    if (summaryPeriod === "daily") {
      const dayStr = format(selected, "yyyy-MM-dd");
      return logs.filter(l => format(new Date(l.timestamp), "yyyy-MM-dd") === dayStr);
    }
    if (summaryPeriod === "weekly") {
      const ws = startOfWeek(selected, { weekStartsOn: 1 });
      const we = endOfWeek(selected, { weekStartsOn: 1 });
      return logs.filter(l => { const d = new Date(l.timestamp); return d >= ws && d <= we; });
    }
    if (summaryPeriod === "monthly") {
      const ms = startOfMonth(selected);
      const me = endOfMonth(selected);
      return logs.filter(l => { const d = new Date(l.timestamp); return d >= ms && d <= me; });
    }
    // custom — date range + optional time window
    if (summaryPeriod === "custom") {
      const rangeStart = new Date(customStartDate + "T00:00:00");
      const rangeEnd   = new Date(customEndDate   + "T23:59:59");
      const [fromH, fromM] = customTimeFrom.split(":").map(Number);
      const [toH,   toM]   = customTimeTo.split(":").map(Number);
      const timeFromSecs = fromH * 3600 + fromM * 60;
      const timeToSecs   = toH   * 3600 + toM   * 60;
      const isFullDay    = timeFromSecs === 0 && timeToSecs >= 23 * 3600 + 59 * 60;
      return logs.filter(l => {
        const d = new Date(l.timestamp);
        if (d < rangeStart || d > rangeEnd) return false;
        if (!isFullDay) {
          const secs = d.getHours() * 3600 + d.getMinutes() * 60 + d.getSeconds();
          if (secs < timeFromSecs || secs > timeToSecs) return false;
        }
        return true;
      });
    }
    return [];
  }

  function groupByDept(periodLogs: MonitoringLog[]): Record<string, MonitoringLog[]> {
    const map: Record<string, MonitoringLog[]> = {};
    for (const l of periodLogs) {
      if (!map[l.department]) map[l.department] = [];
      map[l.department].push(l);
    }
    return map;
  }

  function periodLabel(): string {
    const selected = new Date(summaryDate + "T00:00:00");
    if (summaryPeriod === "daily") return format(selected, "MMMM d, yyyy");
    if (summaryPeriod === "weekly") {
      const ws = startOfWeek(selected, { weekStartsOn: 1 });
      const we = endOfWeek(selected, { weekStartsOn: 1 });
      const weekNum = getWeek(selected, { weekStartsOn: 1 });
      return `Week ${weekNum} \u2014 ${format(ws, "MMM d")} to ${format(we, "MMM d, yyyy")}`;
    }
    if (summaryPeriod === "monthly") return format(selected, "MMMM yyyy");
    if (summaryPeriod === "custom") {
      const s = new Date(customStartDate + "T00:00:00");
      const e = new Date(customEndDate   + "T00:00:00");
      const dateRange = customStartDate === customEndDate
        ? format(s, "MMM d, yyyy")
        : `${format(s, "MMM d")} \u2013 ${format(e, "MMM d, yyyy")}`;
      const isFullDay = customTimeFrom === "00:00" && customTimeTo === "23:59";
      return isFullDay ? dateRange : `${dateRange} · ${customTimeFrom} \u2013 ${customTimeTo}`;
    }
    return "";
  }

  function toggleDept(dept: string) {
    setExpandedDepts(prev => {
      const next = new Set(prev);
      if (next.has(dept)) next.delete(dept);
      else next.add(dept);
      return next;
    });
  }

  // Render a BatStateU-style 2-column monitoring sheet for a department
  function renderMonitoringSheet(deptLogs: MonitoringLog[], dept: string) {
    const sorted = [...deptLogs].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    const TOTAL_SLOTS = 60;
    const COL_SIZE = 30;
    const slots: Array<{ name: string; dateTime: string } | null> = Array(TOTAL_SLOTS).fill(null);
    sorted.forEach((log, i) => {
      if (i < TOTAL_SLOTS) {
        slots[i] = {
          name: log.name,
          dateTime: format(new Date(log.timestamp), "MM/dd/yy hh:mm aa"),
        };
      }
    });

    const leftCol = slots.slice(0, COL_SIZE);
    const rightCol = slots.slice(COL_SIZE, TOTAL_SLOTS);

    return (
      <div className="mt-3 border border-gray-300 rounded-lg overflow-hidden text-xs">
        {/* Sheet header */}
        <div className="bg-gray-100 border-b border-gray-300 px-4 py-2 flex items-center justify-between">
          <div>
            <p className="font-bold text-gray-800 text-sm">{dept}</p>
            <p className="text-gray-500 text-xs">
              BatStateU-REC-ATT-11 · Employee Monitoring Summary · {periodLabel()}
            </p>
          </div>
          <span className="text-xs text-gray-500 font-medium">{sorted.length} record(s)</span>
        </div>
        {/* Two-column table */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="w-8 px-2 py-1.5 text-center text-gray-500 font-semibold border-r border-gray-200">#</th>
                <th className="px-3 py-1.5 text-left text-gray-600 font-semibold border-r border-gray-200">NAME</th>
                <th className="px-3 py-1.5 text-left text-gray-600 font-semibold border-r border-gray-300 w-32">DATE &amp; TIME</th>
                <th className="w-8 px-2 py-1.5 text-center text-gray-500 font-semibold border-r border-gray-200">#</th>
                <th className="px-3 py-1.5 text-left text-gray-600 font-semibold border-r border-gray-200">NAME</th>
                <th className="px-3 py-1.5 text-left text-gray-600 font-semibold w-32">DATE &amp; TIME</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {leftCol.map((left, i) => {
                const right = rightCol[i] ?? null;
                return (
                  <tr key={i} className="hover:bg-blue-50/20">
                    <td className="px-2 py-1 text-center text-gray-400 border-r border-gray-200 font-mono">{i + 1}.</td>
                    <td className="px-3 py-1 border-r border-gray-200 font-medium text-gray-800 truncate max-w-[140px]">
                      {left?.name ?? <span className="text-gray-300">—</span>}
                    </td>
                    <td className="px-3 py-1 border-r border-gray-300 text-gray-600 font-mono whitespace-nowrap">
                      {left?.dateTime ?? ""}
                    </td>
                    <td className="px-2 py-1 text-center text-gray-400 border-r border-gray-200 font-mono">{COL_SIZE + i + 1}.</td>
                    <td className="px-3 py-1 border-r border-gray-200 font-medium text-gray-800 truncate max-w-[140px]">
                      {right?.name ?? <span className="text-gray-300">—</span>}
                    </td>
                    <td className="px-3 py-1 text-gray-600 font-mono whitespace-nowrap">
                      {right?.dateTime ?? ""}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  if (isAdmin && !selectedDept) {
    return (
      <AppLayout>
        <div className="h-[calc(100vh-12rem)] flex flex-col items-center justify-center text-center gap-6">
          <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center">
            <LayoutDashboard className="w-10 h-10 text-primary" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">No Sub-unit Selected</h2>
            <p className="text-gray-500 mt-2 max-w-sm">
              Please go to the Dashboard, choose an office, then select a sub-unit to view its personnel and monitoring data.
            </p>
          </div>
          <button
            onClick={() => setLocation("/dashboard")}
            className="px-6 py-3 bg-primary text-white rounded-xl font-semibold hover:bg-primary/90 transition-colors"
          >
            Go to Dashboard
          </button>
        </div>
      </AppLayout>
    );
  }

  const periodLogs = activeTab === "summary" ? getLogsForPeriod() : [];
  const deptGroups = activeTab === "summary" ? groupByDept(periodLogs) : {};
  const sortedDepts = Object.keys(deptGroups).sort();

  return (
    <AppLayout>
      <div className="flex flex-col gap-4 h-[calc(100vh-10rem)]">

        {/* Header */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Users className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900">
                {isAdmin ? selectedDept : "My Department"} — Staff Monitoring
              </h2>
              <div className="flex items-center gap-2 mt-0.5">
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

        {/* Tabs */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setActiveTab("logs")}
            className={`px-4 py-2 rounded-xl font-semibold text-sm transition-colors ${activeTab === "logs" ? "bg-primary text-white shadow-sm" : "bg-white border border-gray-200 text-gray-600 hover:bg-gray-50"}`}
          >
            <Clock className="inline w-4 h-4 mr-1.5 -mt-0.5" />
            Monitoring Logs
            {activeTab === "logs" && (
              <span className="ml-2 bg-white/20 text-white text-xs px-1.5 py-0.5 rounded-full">{logs.length}</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("roster")}
            className={`px-4 py-2 rounded-xl font-semibold text-sm transition-colors ${activeTab === "roster" ? "bg-primary text-white shadow-sm" : "bg-white border border-gray-200 text-gray-600 hover:bg-gray-50"}`}
          >
            <Users className="inline w-4 h-4 mr-1.5 -mt-0.5" />
            Personnel Roster
            {activeTab === "roster" && (
              <span className="ml-2 bg-white/20 text-white text-xs px-1.5 py-0.5 rounded-full">{personnel.length}</span>
            )}
          </button>
          {isAdmin && (
            <button
              onClick={() => setActiveTab("summary")}
              className={`px-4 py-2 rounded-xl font-semibold text-sm transition-colors ${activeTab === "summary" ? "bg-primary text-white shadow-sm" : "bg-white border border-gray-200 text-gray-600 hover:bg-gray-50"}`}
            >
              <BarChart3 className="inline w-4 h-4 mr-1.5 -mt-0.5" />
              Record Summary
            </button>
          )}
        </div>

        {/* Content */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden flex-1 flex flex-col min-h-0">

          {/* Monitoring Logs Tab */}
          {activeTab === "logs" && (
            <div className="overflow-auto flex-1">
              {logsError ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-gray-400 gap-3">
                  <WifiOff className="w-10 h-10" />
                  <p className="font-medium">Could not load monitoring logs.</p>
                </div>
              ) : filteredLogs.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-gray-400 gap-3">
                  <Clock className="w-10 h-10" />
                  <p className="font-medium text-gray-600">No monitoring logs yet for {isAdmin ? selectedDept : "your department"}.</p>
                  <p className="text-sm text-center max-w-sm">
                    Start the facial recognition service on your local machine to begin logging monitoring events.
                  </p>
                </div>
              ) : (
                <table className="w-full text-left border-collapse">
                  <thead className="bg-gray-50 sticky top-0 z-10 border-b border-gray-200">
                    <tr>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Name</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Employee ID</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Department</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Position</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Date &amp; Time</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Log Type</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {filteredLogs.map((log) => {
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
                          <td className="py-3 px-5 font-mono text-sm text-gray-600">{log.employeeId}</td>
                          <td className="py-3 px-5">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-gray-100 text-gray-800 border border-gray-200">{log.department}</span>
                          </td>
                          <td className="py-3 px-5 text-sm text-gray-700">{log.position || <span className="text-gray-400 italic">—</span>}</td>
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
          )}

          {/* Personnel Roster Tab */}
          {activeTab === "roster" && (
            <div className="overflow-auto flex-1">
              {personnelLoading ? (
                <div className="h-full flex items-center justify-center">
                  <div className="animate-spin w-8 h-8 border-4 border-primary border-t-transparent rounded-full" />
                </div>
              ) : filteredPersonnel.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center p-8 text-gray-400 gap-2">
                  <Users className="w-10 h-10" />
                  <p className="font-medium">No personnel found.</p>
                </div>
              ) : (
                <table className="w-full text-left border-collapse">
                  <thead className="bg-gray-50 sticky top-0 z-10 border-b border-gray-200">
                    <tr>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Name</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Employee ID</th>
                      <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider">Position</th>
                      {isAdmin && <th className="py-3 px-5 font-bold text-gray-600 text-xs uppercase tracking-wider text-center">Actions</th>}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {filteredPersonnel.map((person) => (
                      <tr key={person.id} className="hover:bg-blue-50/40 transition-colors">
                        <td className="py-3 px-5">
                          <div className="flex items-center gap-3">
                            {person.photoUrl ? (
                              <img src={person.photoUrl} alt={person.firstName} className="w-9 h-9 rounded-full object-cover shadow-sm" />
                            ) : (
                              <div className="w-9 h-9 rounded-full bg-primary/10 text-primary flex items-center justify-center font-bold text-sm">
                                {person.firstName.charAt(0)}
                              </div>
                            )}
                            <div>
                              <p className="font-semibold text-gray-900 text-sm">
                                {person.lastName}, {person.firstName} {person.middleInitial ? `${person.middleInitial}.` : ""}
                              </p>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-5 font-mono text-sm text-gray-700">{person.employeeId}</td>
                        <td className="py-3 px-5 text-sm text-gray-700">{person.position}</td>
                        {isAdmin && (
                          <td className="py-3 px-5">
                            <div className="flex items-center justify-center gap-2">
                              <button
                                onClick={() => setLocation(`/register/${person.id}`)}
                                className="p-1.5 rounded-lg text-gray-400 hover:text-amber-600 hover:bg-amber-50 transition-colors"
                                title="Update"
                              >
                                <Pencil className="w-4 h-4" />
                              </button>
                              {deleteConfirm === person.id ? (
                                <>
                                  <button onClick={() => handleDelete(person.id)} className="px-2.5 py-1 bg-red-600 text-white text-xs rounded-lg font-semibold hover:bg-red-700">
                                    Confirm
                                  </button>
                                  <button onClick={() => setDeleteConfirm(null)} className="px-2.5 py-1 bg-gray-200 text-gray-700 text-xs rounded-lg font-semibold hover:bg-gray-300">
                                    Cancel
                                  </button>
                                </>
                              ) : (
                                <button onClick={() => setDeleteConfirm(person.id)} className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors" title="Delete">
                                  <Trash2 className="w-4 h-4" />
                                </button>
                              )}
                            </div>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          )}

          {/* Record Summary Tab */}
          {activeTab === "summary" && isAdmin && (
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              {/* Controls */}
              <div className="flex-shrink-0 border-b border-gray-200 bg-gray-50 px-6 py-4 flex flex-col gap-3">
                {/* Period selector */}
                <div className="flex flex-wrap items-center gap-3">
                  <div className="flex gap-1 bg-white border border-gray-200 rounded-xl p-1">
                    {(["daily", "weekly", "monthly", "custom"] as const).map(p => (
                      <button
                        key={p}
                        onClick={() => setSummaryPeriod(p)}
                        className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-colors capitalize ${summaryPeriod === p ? "bg-primary text-white shadow-sm" : "text-gray-500 hover:text-gray-800"}`}
                      >
                        {p === "custom" ? "Custom" : p}
                      </button>
                    ))}
                  </div>

                  {/* Daily / Weekly / Monthly date picker */}
                  {summaryPeriod !== "custom" && (
                    <div className="flex items-center gap-2">
                      <label className="text-sm font-medium text-gray-600 whitespace-nowrap">
                        {summaryPeriod === "daily" ? "Date" : summaryPeriod === "weekly" ? "Any date in week" : "Any date in month"}:
                      </label>
                      <input
                        type="date"
                        value={summaryDate}
                        onChange={e => setSummaryDate(e.target.value)}
                        className="h-9 px-3 rounded-lg border border-gray-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                    </div>
                  )}
                </div>

                {/* Custom range pickers */}
                {summaryPeriod === "custom" && (
                  <div className="flex flex-wrap items-center gap-4 bg-white border border-gray-200 rounded-xl px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Date</span>
                      <input
                        type="date"
                        value={customStartDate}
                        onChange={e => {
                          setCustomStartDate(e.target.value);
                          if (e.target.value > customEndDate) setCustomEndDate(e.target.value);
                        }}
                        className="h-8 px-2.5 rounded-lg border border-gray-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                      <span className="text-gray-400 text-sm">to</span>
                      <input
                        type="date"
                        value={customEndDate}
                        min={customStartDate}
                        onChange={e => setCustomEndDate(e.target.value)}
                        className="h-8 px-2.5 rounded-lg border border-gray-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Time</span>
                      <input
                        type="time"
                        value={customTimeFrom}
                        onChange={e => setCustomTimeFrom(e.target.value)}
                        className="h-8 px-2.5 rounded-lg border border-gray-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                      <span className="text-gray-400 text-sm">to</span>
                      <input
                        type="time"
                        value={customTimeTo}
                        onChange={e => setCustomTimeTo(e.target.value)}
                        className="h-8 px-2.5 rounded-lg border border-gray-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                      />
                      <button
                        onClick={() => { setCustomTimeFrom("00:00"); setCustomTimeTo("23:59"); }}
                        className="text-xs text-gray-400 hover:text-primary underline transition-colors"
                      >
                        Reset
                      </button>
                    </div>
                  </div>
                )}

                {/* Stats + export row */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-sm text-gray-600 font-medium">
                    <FileText className="inline w-4 h-4 text-gray-400 mr-1 -mt-0.5" />
                    {periodLabel()} · {periodLogs.length} record(s) across {sortedDepts.length} dept(s)
                  </span>
                  {sortedDepts.length > 0 && (
                    <button
                      onClick={async () => {
                        setExportingAll(true);
                        await exportAllDeptsPDF(
                          deptGroups, summaryPeriod, summaryDate,
                          summaryPeriod === "custom" ? customStartDate : undefined,
                          summaryPeriod === "custom" ? customEndDate   : undefined,
                        );
                        setExportingAll(false);
                      }}
                      disabled={exportingAll}
                      className="flex items-center gap-1.5 px-3 py-1.5 bg-primary text-white rounded-lg text-xs font-semibold hover:bg-primary/90 transition-colors disabled:opacity-60 ml-auto"
                    >
                      <Download className="w-3.5 h-3.5" />
                      {exportingAll ? "Generating..." : "Download All Departments"}
                    </button>
                  )}
                </div>
              </div>

              {/* Department Records */}
              <div className="flex-1 overflow-auto p-6">
                {periodLogs.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-64 text-gray-400 gap-3">
                    <BarChart3 className="w-12 h-12 opacity-40" />
                    <p className="font-medium text-gray-600">No monitoring records for this period.</p>
                    <p className="text-sm text-center max-w-sm">Select a different date or period to view archived records.</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {sortedDepts.map(dept => {
                      const deptLogs = deptGroups[dept];
                      const isExpanded = expandedDepts.has(dept);
                      const isExporting = exportingDept === dept;
                      return (
                        <div key={dept} className="border border-gray-200 rounded-xl overflow-hidden">
                          <div className="flex items-center bg-gray-50 hover:bg-gray-100 transition-colors">
                            <button
                              onClick={() => toggleDept(dept)}
                              className="flex-1 flex items-center gap-3 px-5 py-3.5 text-left"
                            >
                              {isExpanded
                                ? <ChevronDown className="w-4 h-4 text-primary flex-shrink-0" />
                                : <ChevronRight className="w-4 h-4 text-gray-400 flex-shrink-0" />}
                              <span className="font-bold text-gray-800">{dept}</span>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary font-semibold">
                                {deptLogs.length} record(s)
                              </span>
                              <span className="text-xs text-gray-500 ml-auto mr-3">
                                {isExpanded ? "Collapse" : "View records"}
                              </span>
                            </button>
                            <button
                              onClick={async (e) => {
                                e.stopPropagation();
                                setExportingDept(dept);
                                await exportDeptPDF(
                                  dept, deptLogs, summaryPeriod, summaryDate,
                                  summaryPeriod === "custom" ? customStartDate : undefined,
                                  summaryPeriod === "custom" ? customEndDate   : undefined,
                                );
                                setExportingDept(null);
                              }}
                              disabled={isExporting || exportingAll}
                              title={`Download PDF for ${dept}`}
                              className="flex items-center gap-1.5 mr-4 px-3 py-1.5 bg-white border border-gray-200 text-gray-600 rounded-lg text-xs font-semibold hover:bg-primary hover:text-white hover:border-primary transition-colors disabled:opacity-50 shadow-sm"
                            >
                              <Download className="w-3.5 h-3.5" />
                              {isExporting ? "Generating..." : "PDF"}
                            </button>
                          </div>
                          {isExpanded && (
                            <div className="px-5 pb-5">
                              {renderMonitoringSheet(deptLogs, dept)}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </AppLayout>
  );
}
