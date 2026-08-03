import React, { createContext, useContext, useState } from "react";

export interface Office {
  name: string;
  code: string;
  slug: string;
  units: string[];
}

export const OFFICES: Office[] = [
  {
    name: "Office of the Chancellor",
    code: "OC",
    slug: "chancellor",
    units: ["Internal Audit", "Quality Assurance Management", "Sustainable Development"],
  },
  {
    name: "Vice Chancellor for Development and External Affairs",
    code: "VCDEA",
    slug: "vcdea",
    units: ["Planning and Development", "External Affairs", "Resource Generations", "ICT Services"],
  },
  {
    name: "Vice Chancellor for Academic Affairs",
    code: "VCAA",
    slug: "vcaa",
    units: [
      "College of Arts and Sciences",
      "College of Accountancy, Business, and Economics",
      "College of Informatics and Computing Sciences",
      "College of Engineering Technology",
      "College of Teacher Education",
      "College of Engineering",
      "Testing and Admission",
      "Culture and Arts",
      "Registration Services",
      "Scholarship and Financial Assistance",
      "Guidance and Counseling",
      "Library Services",
      "Student Organization and Activities",
      "Student Discipline",
      "OJT",
      "National Service Training Program",
    ],
  },
  {
    name: "Vice Chancellor for Administration and Finance",
    code: "VCAF",
    slug: "vcaf",
    units: [
      "Human Resource Management",
      "Records Management",
      "Procurement",
      "Budget",
      "Cashiering/Disbursing",
      "Accounting",
      "Project Facilities and Management",
      "Environment Management Unit",
      "Property and Supply Management Unit",
      "Property and Supply Management",
      "General Services",
    ],
  },
  {
    name: "Vice Chancellor for Research, Development and Extension Services",
    code: "VCRDES",
    slug: "vcrdes",
    units: ["Extension", "Research", "GAD"],
  },
];

// Flat list of all sub-units (used as the "department" value stored in the DB).
export const ALL_UNITS: string[] = OFFICES.flatMap(o => o.units);

// Backwards compatibility export — the staff-monitoring filter still calls this "department",
// but each value is now a sub-unit name belonging to one of the offices above.
export const DEPARTMENTS = ALL_UNITS;
export type Department = string;

// Helpers
export function getOfficeBySlug(slug: string): Office | undefined {
  return OFFICES.find(o => o.slug === slug);
}

export function getOfficeForUnit(unit: string): Office | undefined {
  return OFFICES.find(o => o.units.includes(unit));
}

interface DepartmentContextValue {
  selectedDept: Department | null;
  setSelectedDept: (dept: Department) => void;
  clearDept: () => void;
}

const DepartmentContext = createContext<DepartmentContextValue | null>(null);

const STORAGE_KEY = "bsu_selected_department";

export function DepartmentProvider({ children }: { children: React.ReactNode }) {
  const [selectedDept, setSelectedDeptState] = useState<Department | null>(() => {
    return localStorage.getItem(STORAGE_KEY);
  });

  const setSelectedDept = (dept: Department) => {
    setSelectedDeptState(dept);
    localStorage.setItem(STORAGE_KEY, dept);
  };

  const clearDept = () => {
    setSelectedDeptState(null);
    localStorage.removeItem(STORAGE_KEY);
  };

  return (
    <DepartmentContext.Provider value={{ selectedDept, setSelectedDept, clearDept }}>
      {children}
    </DepartmentContext.Provider>
  );
}

export function useDepartment() {
  const ctx = useContext(DepartmentContext);
  if (!ctx) throw new Error("useDepartment must be used inside DepartmentProvider");
  return ctx;
}
