import { useEffect, useId, useRef, useState } from "react";

import type { Device } from "../api/devices";

export default function DeviceFilter({ devices, value, onChange }: {
  devices: Device[]; value: string; onChange: (deviceId: string) => void;
}) {
  const [search, setSearch] = useState("");
  const menu = useRef<HTMLDetailsElement>(null);
  const trigger = useRef<HTMLElement>(null);
  const searchInput = useRef<HTMLInputElement>(null);
  const labelId = useId();
  const valueId = useId();
  const selected = devices.find((device) => device.device_id === value);
  const query = search.trim().toLocaleLowerCase();
  const matches = devices.filter((device) =>
    `${device.room} ${device.device_id}`.toLocaleLowerCase().includes(query));

  useEffect(() => {
    function closeOutside(event: PointerEvent) {
      if (event.target instanceof Node && menu.current && !menu.current.contains(event.target)) {
        menu.current.open = false;
      }
    }
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, []);

  function choose(deviceId: string) {
    onChange(deviceId);
    if (menu.current) menu.current.open = false;
    trigger.current?.focus();
  }

  return (
    <div className="device-filter-field">
      <p className="device-filter-label" id={labelId}>Filter by device</p>
      <details className="device-filter" ref={menu} onToggle={(event) => {
        if (event.currentTarget.open) searchInput.current?.focus();
        else setSearch("");
      }} onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          if (menu.current) menu.current.open = false;
          trigger.current?.focus();
        }
      }}>
        <summary ref={trigger} aria-labelledby={`${labelId} ${valueId}`}>
          <span id={valueId}>{value ? selected?.room ?? "Unavailable device" : "All devices"}
            {selected && <small>{selected.device_id}</small>}
          </span>
          <span className="filter-chevron" aria-hidden="true">⌄</span>
        </summary>
        <div className="device-filter-menu">
          <label>Search devices
            <input ref={searchInput} type="search" placeholder="Search by room or device name"
              value={search} onChange={(event) => setSearch(event.target.value)} />
          </label>
          <div className="device-filter-options" role="group" aria-label="Choose a device">
            <button type="button" className="device-filter-option" aria-pressed={value === ""}
              onClick={() => choose("")}>All devices</button>
            {matches.map((device) => <button type="button" key={device.device_id}
              className="device-filter-option" aria-pressed={value === device.device_id}
              onClick={() => choose(device.device_id)}>
              <span>{device.room}</span><small>{device.device_id}</small>
            </button>)}
            {matches.length === 0 && <p className="history-caption" role="status">
              {devices.length === 0 ? "No devices linked to your account." : "No matching devices."}
            </p>}
          </div>
        </div>
      </details>
    </div>
  );
}
