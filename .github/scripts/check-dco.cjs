// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Preserve email-validator 2.0.4's syntax and bounds (Unlicense).
const EMAIL = /^[-!#$%&'*+\/0-9=?A-Z^_a-z{|}~](\.?[-!#$%&'*+\/0-9=?A-Z^_a-z\x60{|}~])*@[a-zA-Z0-9](-*\.?[a-zA-Z0-9])*\.[a-zA-Z](-?[a-zA-Z0-9])+$/;

function validEmail(email) {
  if (typeof email !== "string" || email.length > 254 || !EMAIL.test(email)) {
    return false;
  }
  const [local, domain] = email.split("@");
  return local.length <= 64 && domain.split(".").every(part => part.length <= 63);
}

function validateCommits(commits, expectedCount) {
  if (!Array.isArray(commits) || !Number.isInteger(expectedCount) ||
      expectedCount < 1 || commits.length !== expectedCount) {
    return [{sha: "PR", reason: "Cannot validate the complete PR commit list"}];
  }
  const errors = [];
  for (const entry of commits) {
    // Keep the previous check's merge and GitHub Bot exceptions.
    if ((Array.isArray(entry.parents) && entry.parents.length > 1) ||
        entry.author?.type === "Bot") {
      continue;
    }
    const commit = entry.commit;
    const identities = [commit?.author, commit?.committer];
    if (!identities.every(identity => typeof identity?.name === "string" &&
        typeof identity?.email === "string") || typeof commit?.message !== "string") {
      errors.push({sha: entry.sha, reason: "Invalid author or committer metadata"});
      continue;
    }
    const signoffs = [...commit.message.matchAll(/^Signed-off-by: (.*) <(.*)>$/gim)];
    if (signoffs.length === 0) {
      errors.push({sha: entry.sha, reason: "Missing Signed-off-by trailer"});
      continue;
    }
    if (!validEmail(commit.author.email || commit.committer.email)) {
      errors.push({sha: entry.sha, reason: "Invalid commit email"});
      continue;
    }
    const names = new Set(identities.map(identity => identity.name.toLowerCase()));
    const emails = new Set(identities.map(identity => identity.email.toLowerCase()));
    if (!signoffs.some(([, name, email]) =>
      names.has(name.toLowerCase()) && emails.has(email.toLowerCase()))) {
      errors.push({sha: entry.sha, reason: "Signed-off-by does not match author or committer"});
    }
  }
  return errors;
}

async function checkPullRequest({github, context, core}) {
  const pr = context.payload.pull_request;
  if (!pr?.number || !pr.head?.sha) {
    throw new Error("DCO validation requires a pull-request head");
  }
  const args = {...context.repo, pull_number: pr.number};
  const commits = await github.paginate(github.rest.pulls.listCommits, {
    ...args, per_page: 100,
  });
  const {data: current} = await github.rest.pulls.get(args);
  if (current.head.sha !== pr.head.sha || current.commits !== pr.commits) {
    throw new Error("PR changed while its commits were being validated");
  }
  const errors = validateCommits(commits, pr.commits);
  if (errors.length) {
    core.setFailed(errors.map(error => error.sha + ": " + error.reason).join("\n"));
  } else {
    core.info("DCO passed for all " + commits.length + " PR commits");
  }
}

module.exports = {validEmail, validateCommits, checkPullRequest};
