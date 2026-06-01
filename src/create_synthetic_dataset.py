import csv

OUTPUT_FILE = "test_dataset.csv"

BASE_TS = 1720011600000000 

CONFIG = [
    {"ip_src": "10.0.5.0", "subnet_prefix": "193.168.10", "asn": 64501},
    {"ip_src": "10.0.5.0", "subnet_prefix": "193.168.20", "asn": 64501},
    {"ip_src": "10.0.5.0", "subnet_prefix": "193.178.10", "asn": 64501},
    {"ip_src": "10.0.5.0", "subnet_prefix": "193.178.20", "asn": 64501},
    {"ip_src": "10.0.5.1", "subnet_prefix": "192.168.10", "asn": 64501},
    {"ip_src": "10.0.5.1", "subnet_prefix": "192.168.20", "asn": 64501},
    {"ip_src": "10.0.5.1", "subnet_prefix": "192.178.10", "asn": 64501},
    {"ip_src": "10.0.5.1", "subnet_prefix": "192.178.20", "asn": 64501},
    {"ip_src": "10.0.5.2", "subnet_prefix": "192.168.20", "asn": 64502},
    {"ip_src": "10.0.5.3", "subnet_prefix": "192.168.30", "asn": 64503},
    {"ip_src": "10.0.5.4", "subnet_prefix": "192.168.40", "asn": 64504},
]

SERVICES = [
    (6, 80),
    (6, 22),
    (17, 53),
    (6, 25),
]


def encode_service(protocol, port):
    return (protocol << 16) | port


def generate_row(ip_src, subnet_prefix, asn, x):
    ip_dst = f"{subnet_prefix}.{x}"

    timestamp = BASE_TS + x

    ttl = 64 + (x % 32)

    protocol, port = SERVICES[x % len(SERVICES)]
    service = encode_service(protocol, port)

    ip_src_last_octet = int(ip_src.split(".")[-1])

    zmap = 1 if x % 10 == 0 else 0
    mirai = 1 if x % 15 == 0 else 0
    acknowledged_scanner = ip_src_last_octet

    return [
        timestamp,
        ip_src,
        ip_dst,
        service,
        ttl,
        zmap,
        mirai,
        asn,
        acknowledged_scanner
    ]


def generate_dataset():
    with open(OUTPUT_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "ip.src",
            "ip.dst",
            "service",
            "ttl",
            "zmap",
            "mirai",
            "asn",
            "acknowledged_scanner"
        ])

        for config in CONFIG:
            for x in range(1, 10):
                row = generate_row(
                    config["ip_src"],
                    config["subnet_prefix"],
                    config["asn"],
                    x
                )
                writer.writerow(row)


if __name__ == "__main__":
    generate_dataset()
    print(f"Dataset written to {OUTPUT_FILE}")