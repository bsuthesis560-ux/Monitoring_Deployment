import React, { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { AppLayout } from "@/components/layout/AppLayout";
import {
  Crown,
  Globe2,
  GraduationCap,
  Wallet,
  FlaskConical,
  ChevronRight,
  ChevronDown,
  ArrowRight,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { OFFICES, type Office } from "@/contexts/department-context";
import { useDepartment } from "@/contexts/department-context";

interface DeptStat { department: string; total: number; }

const OFFICE_CONFIG: Record<string, { icon: React.ElementType; gradient: string; accent: string; light: string }> = {
  OC:     { icon: Crown,         gradient: "from-red-500 to-red-700",         accent: "bg-red-50 border-red-200 text-red-700",         light: "bg-red-50" },
  VCDEA:  { icon: Globe2,        gradient: "from-amber-500 to-amber-700",     accent: "bg-amber-50 border-amber-200 text-amber-700",   light: "bg-amber-50" },
  VCAA:   { icon: GraduationCap, gradient: "from-blue-500 to-blue-700",       accent: "bg-blue-50 border-blue-200 text-blue-700",      light: "bg-blue-50" },
  VCAF:   { icon: Wallet,        gradient: "from-emerald-500 to-emerald-700", accent: "bg-emerald-50 border-emerald-200 text-emerald-700", light: "bg-emerald-50" },
  VCRDES: { icon: FlaskConical,  gradient: "from-purple-500 to-purple-700",   accent: "bg-purple-50 border-purple-200 text-purple-700", light: "bg-purple-50" },
};

export default function Dashboard() {
  const [, setLocation]   = useLocation();
  const { setSelectedDept } = useDepartment();
  const [stats, setStats] = useState<DeptStat[]>([]);
  const [expandedOffice, setExpandedOffice] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/stats", { credentials: "include" })
      .then(r => r.json())
      .then(data => setStats(Array.isArray(data) ? data : []))
      .catch(() => {});
  }, []);

  const getOfficeTotal = (office: Office) =>
    stats
      .filter(s => office.units.includes(s.department))
      .reduce((sum, s) => sum + (s.total || 0), 0);

  const getUnitTotal = (unit: string) =>
    stats.find(s => s.department === unit)?.total ?? 0;

  const handleOfficeClick = (office: Office) => {
    setExpandedOffice(prev => prev === office.slug ? null : office.slug);
  };

  const handleUnitSelect = (unit: string) => {
    setSelectedDept(unit);
    setLocation("/staff-monitoring");
  };

  return (
    <AppLayout>
      <div className="max-w-5xl mx-auto">
        <div className="mb-8">
          <p className="text-xs font-bold text-gray-500 tracking-[0.25em] uppercase">Offices</p>
          <h1 className="text-3xl font-bold text-gray-900 mt-2">Select an Office</h1>
          <p className="text-gray-500 mt-2">
            Choose an office, then select a sub-unit to view its attendance data.
          </p>
        </div>

        <div className="flex flex-col gap-4">
          {OFFICES.map((office, index) => {
            const cfg      = OFFICE_CONFIG[office.code];
            const Icon     = cfg.icon;
            const total    = getOfficeTotal(office);
            const isOpen   = expandedOffice === office.slug;

            return (
              <motion.div
                key={office.slug}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden"
              >
                {/* Office header row — click to expand */}
                <button
                  onClick={() => handleOfficeClick(office)}
                  className="w-full text-left flex items-center gap-4 p-5 hover:bg-gray-50 transition-colors group"
                >
                  {/* Gradient icon */}
                  <div className={`w-11 h-11 rounded-xl bg-gradient-to-br ${cfg.gradient} flex items-center justify-center shadow-md flex-shrink-0 group-hover:scale-105 transition-transform duration-200`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>

                  {/* Name + code */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="text-base font-bold text-gray-900 truncate">{office.name}</h3>
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold border flex-shrink-0 ${cfg.accent}`}>
                        {office.code}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {office.units.length} sub-units · {total} personnel
                    </p>
                  </div>

                  {/* Chevron */}
                  <motion.div
                    animate={{ rotate: isOpen ? 90 : 0 }}
                    transition={{ duration: 0.2 }}
                    className="flex-shrink-0"
                  >
                    <ChevronRight className="w-5 h-5 text-gray-300 group-hover:text-primary transition-colors" />
                  </motion.div>
                </button>

                {/* Sub-unit dropdown */}
                <AnimatePresence>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.22, ease: "easeInOut" }}
                      className="overflow-hidden"
                    >
                      <div className={`border-t border-gray-100 ${cfg.light} px-5 py-3`}>
                        <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">
                          Select Sub-Unit
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                          {office.units.map(unit => {
                            const unitTotal = getUnitTotal(unit);
                            return (
                              <button
                                key={unit}
                                onClick={() => handleUnitSelect(unit)}
                                className="flex items-center justify-between gap-2 px-4 py-2.5 bg-white rounded-xl border border-gray-200 hover:border-primary hover:bg-primary/5 hover:shadow-sm transition-all text-left group/unit"
                              >
                                <span className="text-sm font-medium text-gray-700 group-hover/unit:text-primary transition-colors truncate">
                                  {unit}
                                </span>
                                <div className="flex items-center gap-1.5 flex-shrink-0">
                                  {unitTotal > 0 && (
                                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${cfg.accent}`}>
                                      {unitTotal}
                                    </span>
                                  )}
                                  <ArrowRight className="w-3.5 h-3.5 text-gray-300 group-hover/unit:text-primary transition-colors" />
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </div>

        <div className="mt-8 bg-white rounded-2xl p-6 border border-gray-100 shadow-sm">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-4">Admin Actions</h3>
          <div className="flex flex-wrap gap-3">
            <motion.a
              href="/register"
              onClick={(e) => { e.preventDefault(); setLocation("/register"); }}
              whileHover={{ scale: 1.02 }}
              className="px-5 py-2.5 bg-primary/10 text-primary hover:bg-primary hover:text-white rounded-xl font-semibold transition-colors text-sm"
            >
              + Register New Personnel
            </motion.a>
            <motion.a
              href="/accounts"
              onClick={(e) => { e.preventDefault(); setLocation("/accounts"); }}
              whileHover={{ scale: 1.02 }}
              className="px-5 py-2.5 bg-secondary/10 text-secondary hover:bg-secondary hover:text-white rounded-xl font-semibold transition-colors text-sm"
            >
              Manage User Accounts
            </motion.a>
            <motion.a
              href="/sample-pdfs"
              onClick={(e) => { e.preventDefault(); setLocation("/sample-pdfs"); }}
              whileHover={{ scale: 1.02 }}
              className="px-5 py-2.5 bg-gray-100 text-gray-600 hover:bg-gray-200 rounded-xl font-semibold transition-colors text-sm"
            >
              Sample PDF Reports
            </motion.a>
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
