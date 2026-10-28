# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Command-line entry point for the standalone KVCR service."""

from .service.server import main

if __name__ == "__main__":
    main()
