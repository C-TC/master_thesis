import os
import time
import socket

node_list = os.getenv("SLURM_JOB_NODELIST")

if node_list is None:
    print("not a slurm job")
    exit(33)

node_hostnames = []
prefix_end = node_list.find("[")
if prefix_end == -1:
    node_hostnames.append(node_list)
else:
    prefix = node_list[:prefix_end]

    for suffix in node_list[prefix_end + 1: -1].split(","):
        if "-" in suffix:
            range_start, range_stop = [int(i) for i in suffix.split("-")]
            node_hostnames.extend(
                f"{prefix}{suf}" for suf in range(range_start, range_stop + 1)
            )
        else:
            node_hostnames.append(prefix + suffix)

hostname = os.getenv("SLURMD_NODENAME")
# only one node writes to a file
if hostname is not None and hostname == node_hostnames[0]:
    node_ips = [
        socket.gethostbyname(hostname)
        for hostname in node_hostnames
    ]
    print(f"Creating a node list in '{os.getcwd()}'. Found {len(node_ips)} nodes.")
    with open("ip_config.txt", "w") as f:
        f.write("\n".join(node_ips))
else:
    time.sleep(1)
