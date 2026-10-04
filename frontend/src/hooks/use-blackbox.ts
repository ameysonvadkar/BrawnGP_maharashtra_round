// Live data hooks shared by the landing page and the debugger.
import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, queries } from "@/lib/api";

export function useUtcClock(): string {
  const [now, setNow] = useState<string>("--:--:--");
  useEffect(() => {
    const tick = () => setNow(new Date().toISOString().slice(11, 19));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}

/** Health with round-trip latency, polled every 10 s. */
export function useLiveStatus() {
  const [latency, setLatency] = useState<number | null>(null);
  const health = useQuery({
    ...queries.health(),
    queryFn: async () => {
      const t0 = performance.now();
      const result = await api.health();
      setLatency(Math.round(performance.now() - t0));
      return result;
    },
  });
  const online = health.isSuccess && health.data.status === "ok";
  return { online, loading: health.isPending, latency, health: health.data };
}
