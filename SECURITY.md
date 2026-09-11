# Security Policy

## Reporting a vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Report suspected vulnerabilities privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository, or by contacting the maintainers directly.

Please include:

- a description of the issue and its potential impact
- the steps required to reproduce it
- the affected version or commit

We will acknowledge receipt and provide an assessment of next steps.

## Scope

This repository contains a research software package for reduced-form modeling.
It is intended for scientific and analytical use rather than production
deployment, and it is not hardened against untrusted input.

In particular, configuration files, model artifacts, and input datasets are
loaded as trusted inputs. Do not run this code against configuration or data
files from an untrusted source. The distributed and HPC execution paths submit
jobs and write to shared filesystems; review configuration before running them
in a shared environment.
