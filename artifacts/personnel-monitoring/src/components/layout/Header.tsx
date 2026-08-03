import React from "react";
import { Link } from "wouter";

export function Header() {
  return (
    <header className="w-full bg-white flex flex-col shadow-sm relative z-10">
      {/* BSU Main Header Area */}
      <div className="flex flex-col items-center justify-center py-8 border-b-[6px] border-primary px-4 text-center">
        <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground font-serif tracking-tight">
          BATANGAS STATE UNIVERSITY
        </h1>
        <h2 className="text-lg md:text-xl font-bold text-primary font-serif mt-1 tracking-wide">
          THE NATIONAL ENGINEERING UNIVERSITY
        </h2>
        <div className="mt-6 inline-block bg-gray-50 px-8 py-3 rounded-full border border-gray-100 shadow-inner">
          <h3 className="text-xl md:text-2xl font-black text-gray-800 tracking-widest">
            PERSONNEL MONITORING SYSTEM
          </h3>
        </div>
      </div>
      
      {/* Institutional Bars */}
      <div className="w-full bg-secondary py-2.5 px-4 text-white text-center font-bold tracking-[0.2em] text-xs md:text-sm shadow-inner">
        LEADING INNOVATIONS, TRANSFORMING LIVES, BUILDING THE NATION
      </div>
      <div className="w-full bg-primary py-2 px-4 text-white text-center font-bold tracking-[0.3em] text-xs md:text-sm shadow-md">
        LIPA CAMPUS
      </div>
    </header>
  );
}
