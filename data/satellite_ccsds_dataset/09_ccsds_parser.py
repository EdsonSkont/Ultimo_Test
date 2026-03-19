#!/usr/bin/env python3
"""
CCSDS Space Packet Parser
Standard: CCSDS 133.0-B-2
Usage: python 09_ccsds_parser.py <binary_file>

Parses TM/TC packets from binary files produced in this dataset.
"""
import struct, sys, json
from pathlib import Path

APID_MAP = {
    0x0C3: "Housekeeping TM",
    0x200: "ADCS Command TC",
    0x301: "Routing Update",
    0x080: "GNSS Ephemeris",
    0x4FF: "SAR Payload",
    0x100: "Timing Sync PPS",
    0x500: "IP Relay User Data",
    0x7FE: "Fault Alert",
}

def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def parse_primary_header(data: bytes) -> dict:
    if len(data) < 6:
        raise ValueError("Packet too short for primary header")
    w0, w1, w2 = struct.unpack(">HHH", data[:6])
    return {
        "version": (w0 >> 13) & 0x07,
        "type": "TC" if (w0 >> 12) & 1 else "TM",
        "secondary_header": bool((w0 >> 11) & 1),
        "apid": w0 & 0x07FF,
        "apid_hex": f"0x{w0 & 0x07FF:03X}",
        "seq_flags": (w1 >> 14) & 0x03,
        "seq_count": w1 & 0x3FFF,
        "data_length": w2 + 1,
        "description": APID_MAP.get(w0 & 0x07FF, "Unknown"),
    }

def decode_tai_timestamp(data: bytes) -> str:
    """Decode 6-byte CCSDS TAI timestamp to UTC string"""
    coarse, fine = struct.unpack(">IH", data[:6])
    # TAI epoch: 2000-01-01 11:58:55.816 UTC
    import datetime
    epoch = datetime.datetime(2000, 1, 1, 11, 58, 55, 816000,
                               tzinfo=datetime.timezone.utc)
    dt = epoch + datetime.timedelta(seconds=coarse + fine/65536)
    return dt.isoformat()

def decode_hk_payload(payload: bytes) -> dict:
    """Decode housekeeping telemetry payload (after 6-byte timestamp)"""
    if len(payload) < 6 + 13:
        return {"raw_hex": payload.hex()}
    ts   = decode_tai_timestamp(payload[:6])
    data = payload[6:]
    try:
        batt, temp, solar, rpm, mode, q1, q2, rssi, fault, orb_hi, cpu = \
            struct.unpack(">HHHHBHHBBBb", data[:17])
        return {
            "timestamp": ts,
            "battery_V": round(batt / 100, 3),
            "panel_temp_C": round((temp - 2731) / 10, 2),
            "solar_mA": solar,
            "rw_rpm": rpm,
            "adcs_mode": ["Coarse","Sun","Fine","Standby"][mode % 4],
            "q1_raw": q1,
            "q2_raw": q2,
            "rssi_dBm": -120 + rssi,
            "fault_flag": f"0x{fault:02X}",
            "orbit_num_hi": orb_hi,
            "cpu_temp_C": round(cpu + 34.5, 1),
        }
    except struct.error:
        return {"raw_hex": data.hex()}

def parse_file(filepath: str, max_packets: int = 50):
    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        return []
    
    results = []
    with open(path, "rb") as f:
        raw = f.read()
    
    offset = 0
    count  = 0
    while offset + 6 <= len(raw) and count < max_packets:
        try:
            hdr = parse_primary_header(raw[offset:])
            total_len = 6 + hdr["data_length"] + 2  # hdr + data + CRC
            if offset + total_len > len(raw):
                break
            packet_bytes = raw[offset:offset + total_len]
            crc_recv = struct.unpack(">H", packet_bytes[-2:])[0]
            crc_calc = crc16_ccitt(packet_bytes[:-2])
            hdr["crc_valid"] = crc_recv == crc_calc
            hdr["crc_received"] = f"0x{crc_recv:04X}"
            hdr["packet_offset"] = offset
            hdr["packet_size_bytes"] = total_len
            if hdr["apid"] == 0x0C3 and hdr["secondary_header"]:
                hdr["decoded"] = decode_hk_payload(packet_bytes[6:-2])
            results.append(hdr)
            offset += total_len
            count  += 1
        except Exception as e:
            offset += 1
    return results

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "01_tm_housekeeping.bin"
    print(f"\nParsing: {target}")
    print("="*60)
    pkts = parse_file(target, max_packets=10)
    for i, p in enumerate(pkts):
        print(f"\nPacket #{i+1} @ offset {p['packet_offset']}:")
        print(f"  Type : {p['type']}  |  APID: {p['apid_hex']} ({p['description']})")
        print(f"  Seq  : {p['seq_count']}  |  Data length: {p['data_length']} B")
        print(f"  CRC  : {p['crc_received']} — {'PASS' if p['crc_valid'] else 'FAIL'}")
        if "decoded" in p:
            d = p["decoded"]
            if "timestamp" in d:
                print(f"  Time : {d['timestamp']}")
            for k, v in list(d.items())[:6]:
                if k != "timestamp":
                    print(f"  {k:20s}: {v}")
    print(f"\nTotal packets parsed: {len(pkts)}")
    print("For JSON output, import this module and call parse_file(path)")
