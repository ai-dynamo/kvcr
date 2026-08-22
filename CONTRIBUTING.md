<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to kvcr

Thank you for your interest in contributing to KVCR (KV Cache Runner)!

KVCR is a component that works with the KV router to simplify the KV cache
management boundary.

**This project accepts external contributions.** Contributions from outside
NVIDIA are welcome and are accepted under the Developer Certificate of Origin
(DCO) sign-off described in [Signing Your Work](#signing-your-work) below. Every
commit in a pull request must carry a `Signed-off-by` line; this is enforced on
every pull request by the `dco` CI workflow.

Before opening a pull request, make sure you can contribute your work to open
source — that your code introduces no license or patent conflict. You must
certify compliance with this project's [license terms](LICENSE) and sign off on
the Developer Certificate of Origin before your pull request can be merged.

## How to Contribute

Standard GitHub PR workflow:

1. Create a topic branch from `main`.
2. Make your changes; keep PRs focused (one logical change per PR).
3. Make sure every new source file carries the SPDX header block (see
   [Source Headers](#source-headers) below).
4. Sign off your commits (see [Signing Your Work](#signing-your-work) below).
5. Open the PR. CI must pass.

For security disclosures, do **not** open a public issue or pull request - see
[SECURITY.md](SECURITY.md).

## Signing Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

- Any contribution which contains commits that are not Signed-Off will not be accepted.
- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO (https://developercertificate.org/):

  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```

## Source Headers

All new source files must carry the SPDX header block. Use the form appropriate for the file type.

**Rust / C-style comments:**

```
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

**Shell / Python / YAML / TOML:**

```
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```

## License

By contributing, you agree that your contributions are licensed under the [Apache 2.0 license](LICENSE) of this project.
