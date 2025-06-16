# High Performance Computing SLURM Course

High Performance Computing (HPC) is an incredibly powerful tool for developers—especially for AI engineers training large models. You can achieve strong AI models through three key ingredients:

1. **Lots of data**  
2. **Powerful compute resources**  
3. **Effective learning algorithms**  

While HPC is extremely useful, many users find that you can run into issues you wouldn’t see on a local machine, and the errors can be harder to debug.

---

## Connectivity

### SSH to IDUN (outside NTNU network)

```bash
ssh -J <username>@login.stud.ntnu.no <username>@idun-login1.hpc.ntnu.no
```

### SSH to IDUN (inside NTNU network)

```bash
ssh <username>@idun-login1.hpc.ntnu.no
```
## Checking your Quota

Check all groups’ quotas:
```bash
idun-slurm-quota
```
Check your own group associations:
```bash
sacctmgr show assoc where user=<username> format=account --parsable
```

## Key SLURM Commands
Submit a job:
```bash
sbatch <path-of-job.slurm>
```

Cancel a job:
```bash
scancel <job-id>
```

List running/pending jobs:
```bash
squeue -u <username>
```



* Most user-friendly tutorial: [IDUN Tutorial from the Perspective of a Master’s Student (2024 edition)](https://www.hpc.ntnu.no/idun/documentation/idun-tutorial-from-the-perspective-of-a-masters-student-2024-edition/)
* Complete SLURM Docs: https://slurm.schedmd.com
* IDUN & Hardware Specs: https://www.hpc.ntnu.no/idun/
* IDUN Status Page (NTNU intranet only): http://idun.hpc.ntnu.no/
* Connect VS Code to IDUN: https://www.hpc.ntnu.no/idun/documentation/visual-studio-code-connect-to-idun-hpc-cluster/
* General Documentation: https://www.hpc.ntnu.no/idun/documentation/