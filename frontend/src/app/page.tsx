import { DottedSurface } from "@/components/ui/dotted-surface";
import { cn } from "@/lib/utils";

export default function Home() {
  return (
    <>
      {/* Animated dotted wave background */}
      <DottedSurface />

      {/* Radial glow overlay */}
      <div
        aria-hidden="true"
        className={cn(
          "pointer-events-none fixed inset-0 -z-10",
          "bg-[radial-gradient(ellipse_80%_60%_at_50%_100%,rgba(6,150,215,0.08),transparent)]"
        )}
      />

      {/* Page content */}
      <main className="relative flex flex-1 flex-col items-center justify-center min-h-screen px-6 text-center">
        {/* Top badge */}
        <span className="mb-6 inline-flex items-center gap-2 rounded-full border border-blue-200 bg-blue-50/80 px-4 py-1.5 text-xs font-semibold uppercase tracking-widest text-blue-700 backdrop-blur-sm dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300">
          ISA S5.1 &middot; ISO 10628
        </span>

        {/* Heading */}
        <h1 className="max-w-3xl text-5xl font-bold tracking-tight text-[#0a1628] dark:text-white sm:text-6xl">
          P&amp;ID Intelligence{" "}
          <span className="text-[#0696D7]">Platform</span>
        </h1>

        <p className="mt-6 max-w-xl text-lg leading-relaxed text-slate-600 dark:text-slate-400">
          Automated symbol detection, OCR extraction, knowledge graph
          construction, and AI-powered engineering assistance for
          Piping &amp; Instrumentation Diagrams.
        </p>

        {/* CTA buttons */}
        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <a
            href="http://localhost:8501"
            className="inline-flex h-12 items-center justify-center rounded-full bg-[#0a1628] px-8 text-sm font-semibold text-white shadow-md transition-all hover:bg-[#0696D7] hover:shadow-blue-200 dark:hover:shadow-blue-900"
          >
            Open Platform
          </a>
          <a
            href="https://github.com"
            className="inline-flex h-12 items-center justify-center rounded-full border border-slate-300 bg-white/70 px-8 text-sm font-semibold text-slate-800 backdrop-blur-sm transition-all hover:border-blue-300 hover:bg-white dark:border-slate-700 dark:bg-slate-900/60 dark:text-slate-200 dark:hover:border-blue-600"
          >
            View Source
          </a>
        </div>

        {/* Feature pills */}
        <div className="mt-16 flex flex-wrap justify-center gap-3 text-sm font-medium">
          {[
            "YOLO Symbol Detection",
            "PaddleOCR Text Extraction",
            "NetworkX Knowledge Graph",
            "Claude AI Assistant",
          ].map((feat) => (
            <span
              key={feat}
              className="rounded-lg border border-slate-200 bg-white/60 px-4 py-2 text-slate-700 backdrop-blur-sm dark:border-slate-700 dark:bg-slate-900/50 dark:text-slate-300"
            >
              {feat}
            </span>
          ))}
        </div>
      </main>

      {/* Footer */}
      <footer className="relative py-6 text-center text-xs text-slate-400 dark:text-slate-600">
        P&amp;ID Intelligence Platform &mdash; v1.0.0
      </footer>
    </>
  );
}
