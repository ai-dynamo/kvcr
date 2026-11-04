// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const assert = require("node:assert/strict");
const {test} = require("node:test");
const {validEmail, validateCommits, checkPullRequest} = require("./check-dco.cjs");

const SIGNOFF = "Signed-off-by: Example Author <author@example.org>";
function commit(options = {}) {
  return {
    sha: "a".repeat(40),
    parents: options.merge ? [{}, {}] : [{}],
    author: options.githubAuthor === undefined
      ? {type: "User", login: "contributor"} : options.githubAuthor,
    commit: {
      author: {name: "Example Author", email: options.email ?? "author@example.org"},
      committer: {name: "Example Committer", email: "committer@example.org"},
      message: options.message ?? "Example change\n\n" + SIGNOFF,
    },
  };
}

const accepted = [
  ["author", {}],
  ["committer", {message: "Signed-off-by: Example Committer <committer@example.org>"}],
  ["case-insensitive", {message: "signed-OFF-by: EXAMPLE AUTHOR <AUTHOR@EXAMPLE.ORG>"}],
  ["multiple trailers", {message: "Signed-off-by: Other <other@example.org>\n" + SIGNOFF}],
  ["unmapped GitHub author", {githubAuthor: null}],
  ["signed author with coauthor", {message: SIGNOFF + "\nCo-authored-by: Other <other@example.org>"}],
  ["existing independent name/email matching", {
    message: "Signed-off-by: Example Author <committer@example.org>",
  }],
  ["merge exception", {merge: true, message: "Merge branch"}],
  ["GitHub Bot exception", {githubAuthor: {type: "Bot"}, message: "Automated change"}],
];
for (const [name, options] of accepted) {
  test("accepts " + name, () => assert.deepEqual(validateCommits([commit(options)], 1), []));
}

const rejected = [
  ["unsigned", {message: "Example change"}],
  ["wrong name", {message: "Signed-off-by: Other <author@example.org>"}],
  ["wrong email", {message: "Signed-off-by: Example Author <other@example.org>"}],
  ["coauthor without signoff", {message: "Co-authored-by: Other <other@example.org>"}],
  ["coauthor signoff cannot replace author", {message: "Signed-off-by: Other <other@example.org>"}],
  ["unmapped unsigned author", {githubAuthor: null, message: "Example change"}],
  ["invalid commit email", {email: "invalid", message: "Signed-off-by: Example Author <invalid>"}],
  ["malformed trailer", {message: "Signed-off-by:Example Author <author@example.org>"}],
];
for (const [name, options] of rejected) {
  test("rejects " + name, () => assert.equal(validateCommits([commit(options)], 1).length, 1));
}

test("rejects malformed identity metadata", () => {
  const entry = commit();
  entry.commit.author = null;
  assert.equal(validateCommits([entry], 1).length, 1);
});

test("fails closed on missing, incomplete, or extra commit records", () => {
  for (const [entries, count] of [[[], 1], [[commit()], 2], [[commit()], 0],
      [[commit()], undefined], [[commit(), commit()], 1], [null, 1]]) {
    assert.equal(validateCommits(entries, count).length, 1);
  }
});

test("retains email syntax and length bounds", () => {
  for (const address of ["a..b@example.org", ".a@example.org", "a@localhost",
      "a@bad_domain.org", "a".repeat(65) + "@example.org",
      "a@" + "x".repeat(64) + ".org"]) {
    assert.equal(validEmail(address), false, address);
  }
  assert.equal(validEmail("author+tag@example.org"), true);
});

function apiFixture(entries, count = entries.length) {
  const calls = [];
  const context = {
    repo: {owner: "ai-dynamo", repo: "kvcr"},
    payload: {pull_request: {number: 104, commits: count, head: {sha: "expected"}}},
  };
  const current = {head: {sha: "expected"}, commits: count};
  const failures = [];
  return {
    context, current, calls, failures,
    core: {setFailed: message => failures.push(message), info: () => {}},
    github: {
      rest: {pulls: {listCommits: {}, get: async () => ({data: current})}},
      paginate: async (route, args) => {
        calls.push(args);
        return entries;
      },
    },
  };
}

test("checks every paginated commit, including an unsigned final page", async () => {
  const entries = Array.from({length: 101}, () => commit());
  entries[100] = commit({message: "Unsigned final page"});
  const fixture = apiFixture(entries);
  await checkPullRequest(fixture);
  assert.equal(fixture.calls[0].per_page, 100);
  assert.equal(fixture.failures.length, 1);
  assert.match(fixture.failures[0], /Missing Signed-off-by/);
});

test("rejects API truncation rather than passing unchecked commits", async () => {
  const fixture = apiFixture([commit()], 251);
  await checkPullRequest(fixture);
  assert.match(fixture.failures[0], /complete PR commit list/);
});

test("passes a complete valid pull request", async () => {
  const fixture = apiFixture([commit()]);
  await checkPullRequest(fixture);
  assert.deepEqual(fixture.failures, []);
});

test("rejects head or count changes during pagination", async () => {
  for (const field of ["head", "commits"]) {
    const fixture = apiFixture([commit()]);
    if (field === "head") fixture.current.head.sha = "changed";
    else fixture.current.commits = 2;
    await assert.rejects(checkPullRequest(fixture), /PR changed/);
  }
});

test("requires pull-request context", async () => {
  const fixture = apiFixture([commit()]);
  fixture.context.payload = {};
  await assert.rejects(checkPullRequest(fixture), /pull-request head/);
});
