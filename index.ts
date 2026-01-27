import assert from 'assert';
import { SHA1 } from 'bun';
import prand from 'pure-rand';

import { Database as Sqlite } from "bun:sqlite"


export type Nullable<T> = T | null;
export type Maybe<T> = T | undefined;

// Attempted to make choices on a test case that has been completed
class Frozen extends Error {
  public override name = 'Frozen';
}

// Raised when a test should stop executing early
class StopTest extends Error {
  public override name = 'StopTest';
}

// Raised when a test has no valid exampples
class Unsatisfiable extends Error {
  public override name = 'Unsatisfiable';
}

class ValueError extends Error {
  public override name = 'ValueError';
  constructor(message: string) {
    super(message);
  }
}

export const Errors = {
  Frozen,
  StopTest,
  Unsatisfiable,
  ValueError
} as const;

class BigintArray {
  private data: BigInt64Array;
  private count: number = 0;

  // Make it indexable
  [idex: number]: bigint;

  static from(data: BigInt64Array): BigintArray {
    const arr = new BigintArray(data.length);
    arr.data.set(data);
    return arr;
  }

  constructor(length: number = 16) {
    this.data = new BigInt64Array(length);

    // Proxy makes it indexable
    return new Proxy(this, {
      get(target: BigintArray, prop: string | symbol): unknown {
        if (typeof prop === 'string' && !isNaN(Number(prop))) {
          return target.get(Number(prop));
        }
        return Reflect.get(target, prop);
      },
      set(target: BigintArray, prop: string | symbol, value: unknown): boolean {
        if (typeof prop === 'string' && !isNaN(Number(prop))) {
          target.put(Number(prop), value as bigint);
          return true;
        }
        return Reflect.set(target, prop, value);
      }
    });
  }

  public push(value: bigint): void {
    if (this.count === this.data.length) {
      const newData = new BigInt64Array(this.data.length * 2);
      newData.set(this.data);
      this.data = newData;
    }

    this.data[this.count++] = value;
  }

  public concat(other: BigInt64Array): BigintArray {
    const newLength = this.data.length + other.length;
    const newData = new BigInt64Array(newLength);
    newData.set(this.data.subarray(0, this.count), 0);
    newData.set(other, this.count);
    this.data = newData;
    return this;
  }

  public view(): BigInt64Array {
    return this.data.subarray(0, this.count);
  }

  public get length(): number {
    return this.count;
  }

  public get(idx: number): bigint {
    if (idx < 0 || idx >= this.count) {
      throw new RangeError(
        `Index ${idx} out of bounds for BigintArray of length ${this.count}`
      );
    }
    return this.data[idx]!;
  }

  public put(idx: number, value: bigint): void {
    if (idx < 0 || idx >= this.count) {
      throw new RangeError(
        `Index ${idx} out of bounds for BigintArray of length ${this.count}`
      );
    }
    this.data[idx] = value;
  }
}

export interface Random {
  random(): number;
  randint(a: bigint, b: bigint): bigint;
  randrange(a: number, b: number): number;
}

// Using sensible defaults suggested in pure-rand documentation
// https://github.com/dubzzz/pure-rand?tab=readme-ov-file#documentation
export class Rng implements Random {
  private seed = Date.now() ^ (Math.random() * 0x100000000);
  private prng: prand.RandomGenerator;

  constructor(seed?: Maybe<number>) {
    if (Number.isFinite(seed)) {
      this.seed = seed!;
    }
    this.prng = prand.xorshift128plus(this.seed);
  }

  public random(): number {
    // https://github.com/dubzzz/pure-rand?tab=readme-ov-file#generate-32-bit-floating-point-numbers
    const g1 = prand.unsafeUniformIntDistribution(0, (1 << 24) - 1, this.prng);
    const value = g1 / (1 << 24);
    return value;
  }

  // random bigint between [a, b] - both ends inclusive
  public randint(a: bigint, b: bigint): bigint {
    return prand.unsafeUniformBigIntDistribution(a, b, this.prng);
  }

  // random number between [a, b) - right end excluded
  public randrange(a: number, b: number): number {
    return prand.unsafeUniformIntDistribution(a, b - 1, this.prng);
  }
}

export interface Database {
  set(key: string, value: Blob): Promise<void>;
  get(key: string): Maybe<Blob>;
  del(key: string): void;
}

// Lean on bun's built-in sqlite support for a quick key-value database
export class KvStore implements Database {
  static hashKey(key: string): string {
    const hasher = new SHA1();
    hasher.update(key);
    return hasher.digest("hex").slice(0, 10);
  }

  private db: Sqlite;

  constructor(filename: string) {
    this.db = new Sqlite(filename);

    this.db.run(`
      CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY,
        value BLOB NOT NULL
      );
    `);
  }

  public get(key: string): Maybe<Blob> {
    const keyHash = KvStore.hashKey(key);
    const result = this.db.query(
      'SELECT value FROM kv WHERE key = ?'
    ).get(keyHash) as Nullable<{ value: Uint8Array }>;

    if (!result) {
      return;
    }

    return new Blob([result.value]);
  }

  public async set(key: string, value: Blob): Promise<void> {
    const keyHash = KvStore.hashKey(key);
    this.db.run(`
      INSERT OR REPLACE INTO kv (key, value)
      VALUES (? ,?);
    `, [keyHash, await value.bytes()]);
  }

  public del(key: string): void {
    const keyHash = KvStore.hashKey(key);
    this.db.run(`
      DELETE FROM kv WHERE key = ?;
    `, [keyHash]);
  }
}



export enum Status {
  // Test case didn't have enough data to complete
  OVERRUN,
  // Test case contained something that prevented completion
  INVALID,
  // Test case completed just fine but was boring
  VALID,
  // Test case completed and was interesting
  INTERESTING
}

export type RunTestInit = {
  maxExamples: number;
  quiet: boolean;
} & Partial<HasRng & HasDb>;

export function runTest(
  name: string,
  test: (tc: TestCase) => void,
  options: RunTestInit = {
    maxExamples: 100,
    quiet: false,
  }
): void {
  const markFailuresInteresting = (testCase: TestCase): void => {
    try {
      test(testCase);
    } catch (error) {
      if (!!testCase.status) {
        throw error;
      }
      testCase.markStatus(Status.INTERESTING);
    }
  };

  const state = new TestingState(
    options.random ?? new Rng(),
    markFailuresInteresting,
    options.maxExamples
  );

  const db: Database = options.database ?? new KvStore('.minithesis-cache');

  const previousFailure = db.get(name);

  if (!!previousFailure) {

  }

  if (!state.result) {
    state.run();
  }

  if (state.validTestCases === 0) {
    throw new Errors.Unsatisfiable();
  }

  if (!state.result) {
    try {
      db.del(name);
    } catch (error) { }
  } else {
    db.set(name, new Blob([state.result.view()]));
  }

  if (!!state.result) {
    test(TestCase.ForChoices(state.result, !options.quiet));
  }
}

export interface HasSizeLimit {
  maxSize: number;
}

export interface HasRng {
  random: Random;
}

export interface HasDb {
  database: Database;
}

export interface HasName {
  name: string;
}

export type TestCaseInit = {
  prefix: BigintArray;
} & Partial<HasSizeLimit & HasRng>;

// A single generated test case which consists of an 
// underlying set of choices that produce possibilities.
export class TestCase {

  public prefix: BigintArray = new BigintArray();
  public targetingScore?: Maybe<number> = undefined;
  public status?: Maybe<Status> = undefined;
  public choices: BigintArray = new BigintArray();

  private random: Random;
  private maxSize: number = Infinity;
  private printResults: boolean = false;
  private depth: number = 0;

  public static ForChoices(choices: BigintArray, printResults: boolean = false): TestCase {
    return new TestCase({ prefix: choices }, printResults);
  }

  constructor(options: TestCaseInit, printResults = false) {
    this.prefix = options.prefix;
    this.random = options.random ?? new Rng();
    this.printResults = printResults;
    if (Number.isFinite(options.maxSize)) {
      this.maxSize = options.maxSize!;
    }
  }

  // Returns a number in the range [0, n]
  public choice(n: bigint): bigint {
    const result = this.makeChoice(n, () => this.random.randint(0n, n));
    if (this.shouldPrint()) {
      console.log(`choice(${n}): ${result}`);
    }
    return result;
  }

  // Retrun true with probability ``p``
  public weighted(p: number): bigint {
    let result: bigint;
    if (p <= 0) {
      result = this.forcedChoice(0n);
    } else if (p >= 1) {
      result = this.forcedChoice(1n);
    } else {
      result = this.makeChoice(1n, () => this.random.random() <= p ? 1n : 0n);
    }

    if (this.shouldPrint()) {
      console.log(`weighted(${p}): ${result}`);
    }

    return result;
  }

  // Inserts a fake choice into the choice sequence, as if some call to choice() had returned ``n``
  // You almost never need this, but sometimes it can be a useful hint to the shrinker.
  public forcedChoice(n: bigint): bigint {
    if (!isValidUint64(n)) {
      throw new Errors.ValueError(
        `forcedChoice(${n}): n must is not valid uint64]`
      );
    }

    if (this.status) {
      throw new Errors.Frozen();
    }

    if (this.choice.length >= this.maxSize) {
      this.markStatus(Status.OVERRUN);
    }

    this.choices.push(n);
    return n;
  }

  // Mark this test case as invalid
  public reject(): never {
    this.markStatus(Status.INVALID);
  }

  // If the precondition is false, abort the test and mark this test case as invalid
  public assume(precondition: boolean): void | never {
    if (!precondition) {
      this.reject();
    }
  }

  // Set a score to maximize. Multiple calls to this function
  // will override previous ones.
  // 
  // The name and idea come from Löscher, Andreas, and Konstantinos
  // Sagonas. "Targeted property-based testing." ISSTA. 2017, but
  // the implementation is based on that found in Hypothesis,
  // which is not that similar to anything described in the paper.
  public target(score: number): void {
    this.targetingScore = score;
  }

  // Return a possible value from ``possibility``
  public any<U>(possibility: Possibility<U>): U {
    let result: U;
    try {
      this.depth += 1;
      result = possibility.produce(this);
    } finally {
      this.depth -= 1;
    }

    if (this.shouldPrint()) {
      console.log(`any(${possibility}): ${result}`);
    }

    return result;
  }

  // Set the status for this case and throw StopTest
  public markStatus(status: Status): never {
    if (this.status) {
      throw new Errors.Frozen();
    }

    this.status = status;
    throw new Errors.StopTest();
  }

  private shouldPrint(): boolean {
    return this.printResults && this.depth === 0;
  }

  // Make a choice in [0, n], by calling rndMethod if randomness is needed
  private makeChoice(n: bigint, rndMethod: () => bigint): bigint | never {
    if (!isValidUint64(n)) {
      throw new Errors.ValueError(`forcedChoice(${n}): n must be valid uint64`);
    }

    if (this.status) {
      throw new Errors.Frozen();
    }

    if (this.choice.length >= this.maxSize) {
      this.markStatus(Status.OVERRUN);
    }

    let result: bigint;
    if (this.choices.length < this.prefix.length) {
      result = this.prefix[this.choices.length]!;
    } else {
      result = rndMethod();
    }

    this.choices.push(n);
    if (result > n) {
      this.markStatus(Status.INVALID);
    }

    return result;
  }
}

// Represents some range of values that might be used in
// a test, that can be requested from a ``TestCase``.
//
// Pass one of these to TestCase.any to get a concrete value.
export class Possibility<T> {
  public produce: (testCase: TestCase) => T;
  public name: string;

  constructor(produce: (testCase: TestCase) => T, name?: Maybe<string>) {
    this.produce = produce;
    if (name) {
      this.name = name;
    } else {
      this.name = produce.name || 'noname';
    }
  }

  public toString(): string {
    return this.name;
  }

  // Returns a ``Possibility`` where values come from applying 
  // ``f`` to some possible value for ``self``
  public map<S>(f: (x: T) => S): Possibility<S> {
    return new Possibility<S>(
      (testCase: TestCase): S => f(testCase.any(this)),
      `${this.name}.map(${f.name})`
    );
  }

  /// Returns a ``Possibility`` where values come from
  /// applying ``f`` (which should return a new ``Possibility``
  /// to some possible value for ``self`` then returning a possible
  /// value from that.
  public bind<S>(f: (x: T) => Possibility<S>): Possibility<S> {
    const produce = (testCase: TestCase): S =>
      testCase.any(f(testCase.any(this)));

    return new Possibility<S>(
      produce,
      `${this.name}.bind(${f.name})`
    );
  }

  // Returns a ``Possibility`` whose values are any possible
  // value of ``self`` for which ``f`` returns True
  public satisfying(f: (x: T) => boolean): Possibility<T> {
    const produce = (testCase: TestCase): T => {
      for (let i = 0; i < 3; i++) {
        const candidate = testCase.any(this);
        if (f(candidate)) {
          return candidate;
        }
      }
      testCase.reject();
    };

    return new Possibility<T>(
      produce,
      `${this.name}.select(${f.name})`
    );
  }
}

// Any integer in the range [m, n] is possible
export function integers(m: bigint, n: bigint): Possibility<bigint> {
  return new Possibility<bigint>(
    (tc: TestCase): bigint => m + tc.choice(n - m),
    `integers(${m}, ${n})`
  );
}

// Any lists whose elements are possible values from ``elements`` are possible
export function lists<U>(
  elements: Possibility<U>,
  minSize: number = 0,
  maxSize: number = Infinity,
): Possibility<U[]> {
  const produce = (tc: TestCase): U[] => {
    const result: U[] = [];
    while (true) {
      if (result.length < minSize) {
        tc.forcedChoice(1n);
      } else if (result.length + 1 >= Number(maxSize)) {
        tc.forcedChoice(0n);
        break
      } else if (!tc.weighted(0.9)) {
        break
      }
      result.push(tc.any(elements));
    }
    return result;
  };

  return new Possibility<U[]>(
    produce,
    `lists(${elements.name})`
  );
}

// Only ``value`` is possible
export function just<U>(value: U): Possibility<U> {
  return new Possibility<U>(
    (_: TestCase) => value, `just(${value})`
  );
}

// No possible values - test case always rejects
export function nothing(): Possibility<never> {
  const produce = (tc: TestCase): never => {
    tc.reject();
  };

  return new Possibility<never>(produce, `nothing()`);
}

// Possible values can be any value possible for one of ``possibilities``
export function mix_of<T>(...possibilities: Possibility<T>[]): Possibility<T> {
  if (!possibilities?.length) {
    return nothing();
  }

  return new Possibility<T>(
    (tc: TestCase): T =>
      tc.any(possibilities[Number(tc.choice(BigInt(possibilities.length)))]!),
    `mix_of(${possibilities.map(p => p.name).join(', ')})`
  );
}

// TODO@iangregson :: does this work / make sense when ts doesn't have a tuple type?
// Any tuple t of of length len(possibilities) such that t[i] is possible
// for possibilities[i] is possible.
export function tuples(...possibilities: Possibility<any>[]): Possibility<any> {
  return new Possibility<any>(
    (tc: TestCase): any[] => possibilities.map(p => tc.any(p)),
    `tuples(${possibilities.map(p => p.name).join(', ')})`
  );
}


// Tree nodes are either a point at which a choice occurs
// in which case they map the result of the choice to the
// tree node we are in after, or a Status object indicating
// markStatus was called at this point and all future
// choices are irrelevant.
//
// Note that a better implementation of this would use
// a Patricia trie, which implements long non-branching
// paths as an array inline. For simplicity we don't
// do that here.
// XXX The Tree type is recursive
export type Tree = Map<bigint, Status | Map<bigint, unknown>>;

// We cap the maximum amount of entropy a test case can use.
// This prevents cases where the generated test case size explodes
// by effectively rejection
const BUFFER_SIZE = 8 * 1024;

// Returns a cached version of a function that maps
// a choice sequence to the status of calling a test function
// on a test case populated with it. Is able to take advantage
// of the structure of the test function to predict the result
// even if exact sequence of choices has not been seen
// previously.
//
// You can safely omit implementing this at the cost of
// somewhat increased shrinking time.
export class CachedTestFunction {


  private tree: Tree = new Map();
  private testFunction: (testCase: TestCase) => void;

  constructor(testFunction: (testCase: TestCase) => void) {
    this.testFunction = testFunction;
  }

  public call(choices: BigintArray): Status {
    // XXX The type of node is problematic
    let node: any = this.tree;
    try {
      for (const c of choices.view()) {
        node = node[Number(c)];
        // markStatus was called at this point so future choices don't matter
        switch (node) {
          case
            Status.INTERESTING,
            Status.VALID,
            Status.INVALID:
            return node;
          case Status.OVERRUN:
            assert(false, 'Test case overran');
        }
      }
      // If we never enetered an unkown region of the tree, or hti a Status, then 
      // we know that another choice will be made next and the result will overrun.
      return Status.OVERRUN;
    } catch (error) { }

    // Now we have to actually call the test function to find out what happens
    const testCase = TestCase.ForChoices(choices);
    this.testFunction(testCase);
    assert(testCase.status !== undefined, 'Test function did not mark status');

    // We enter the choices made in a tree.
    node = this.tree;
    testCase.choices.view().forEach((c, i) => {
      if (
        BigInt(i) + 1n < testCase.choices.length
        || testCase.status === Status.OVERRUN
      ) {
        try {
          node = node.get(c);
        } catch (error) {
          node.set(c, new Map());
        }
      } else {
        node.set(c, testCase.status);
      }
    });
    return testCase.status!;
  }

}

// We need a custom SortKey type to replicate Python's ability to
// use tuples as sort keys
export type SortKey = [number, BigintArray];

export class TestingState {
  private random: Random;
  private _testFunction: (testCase: TestCase) => void;
  private maxExamples: number;
  private calls: number = 0;
  public validTestCases: number = 0;
  public result?: Maybe<BigintArray>;
  public bestScoring?: Maybe<{
    best: number;
    choices: BigintArray;
  }>;
  public testIsTrivial: boolean = false;

  constructor(
    random: Random,
    testFunction: (testCase: TestCase) => void,
    maxExamples: number,
  ) {
    this.random = random;
    this.maxExamples = maxExamples;
    this._testFunction = testFunction;
  }

  // Returns a key that can be used for the shrinking order of test cases
  private sortKey(choices: BigintArray): SortKey {
    return [choices.length, choices];
  }

  public testFunction(testCase: TestCase): void {
    try {
      this._testFunction(testCase);
    } catch (error) { }

    if (!testCase.status) {
      testCase.status = Status.VALID;
    }

    this.calls += 1;
    if (
      testCase.status >= Status.INVALID
      && testCase.choices.length === 0
    ) {
      this.testIsTrivial = true;
    }

    if (testCase.status >= Status.VALID) {
      this.validTestCases += 1;

      if (!!testCase.targetingScore) {
        let relevantInfo = {
          best: testCase.targetingScore,
          choices: testCase.choices
        };

        if (!this.bestScoring) {
          this.bestScoring = relevantInfo;
        } else {
          const { best } = relevantInfo;
          if (
            testCase.targetingScore &&
            testCase.targetingScore > best
          ) {
            this.bestScoring = relevantInfo;
          }
        }
      }
    }

    if (
      testCase.status === Status.INTERESTING &&
      (!this.result || this.sortKey(testCase.choices) < this.sortKey(this.result))
    ) {
      this.result = testCase.choices;
    }
  }

  // In any test cases have had ``target()`` called on them, do a simple hill-climb
  // algorithm to try and optimise that target score.
  public target(): void {
    if (!!this.result || !this.bestScoring) {
      return;
    }

    // Can we improve by changing choices[i] by ``step``?
    const adjust = (i: number, step: bigint): boolean => {
      assert(!!this.bestScoring, "no best scoring info");
      const { best: score, choices } = this.bestScoring;
      if (choices[i]! + step < 0 || !isValidUint64(choices[i]!)) {
        return false;
      }
      let attempt = choices.view()
      attempt[i]! += step;
      const testCase = new TestCase({
        prefix: BigintArray.from(attempt),
        random: this.random,
        maxSize: BUFFER_SIZE,
      });
      this.testFunction(testCase);
      assert(!!testCase.status, "test case should have been resolved to a status");
      return (
        testCase.status >= Status.VALID &&
        !!testCase.targetingScore &&
        testCase.targetingScore > score
      );
    };

    while (this.shouldKeepGenerating()) {
      const i = this.random.randrange(0, this.bestScoring.choices.length);
      let sign = 0n;

      for (const k of [1n, -1n]) {
        if (!this.shouldKeepGenerating()) {
          return;
        }

        if (adjust(i, k)) {
          sign = k;
          break;
        }
      }

      if (sign === 0n) {
        continue;
      }

      let k = 1n;
      while (this.shouldKeepGenerating() && adjust(i, sign * k)) {
        k *= 2n;
      }
      while (k > 0) {
        while (this.shouldKeepGenerating() && adjust(i, sign * k)) { }
        k /= 2n;
      }
    }
  }

  public run(): void {
    this.generate();
    this.target();
    this.shrink();
  }

  private shouldKeepGenerating(): boolean {
    return (
      !this.testIsTrivial &&
      !this.result &&
      this.validTestCases < this.maxExamples &&

      // We impose a limit on the maximum number of calls as
      // well as the maximum number of valid examples. This is
      // to avoid taking a prohibitively long time on tests which
      // have hard or impossible to satisfy preconditions.
      this.calls < this.maxExamples * 10
    );
  }

  // Run random generation until either we have found an interesting test case or 
  // hit the limit of how many test cases we should evaluate.
  private generate(): void {
    while (this.shouldKeepGenerating() && (
      !this.bestScoring || this.validTestCases < Math.floor(this.maxExamples / 2)
    )) {
      this.testFunction(
        new TestCase({ prefix: new BigintArray(), random: this.random, maxSize: BUFFER_SIZE })
      )
    }
  }

  // If we have found an interesting example, try shrinking it
  // so that the choice sequence leading to our best example is
  // shortlex smaller than the one we originally found. This improves
  // the quality of the generated test case, as per our paper.

  // https://drmaciver.github.io/papers/reduction-via-generation-preview.pdf
  private shrink(): void {
    if (!this.result) {
      return;
    }

    // Shrinking will typically try the same choice sequences many many times,
    // so we use a cached version of the test function to avoid wasted cycles.
    // This also allows us to ignore cases where we try e.g. a prefix of choices
    // that is guaranteed not to work.
    let cached = new CachedTestFunction(this._testFunction);

    const consider = (choices: BigintArray): boolean => {
      if (choices == this.result) {
        return true;
      }

      return cached.call(choices) === Status.INTERESTING;
    };

    assert(consider(this.result));

    // We are going to perform a number of transformations to the cyrrent result, 
    // iterating until none of them make any progress - i.e. until we make it through 
    // an entire iteration of the loop without changing the result.
    let prev = null;
    while (prev !== this.result) {
      prev = this.result;

      // A note on weird loop order: We iterate backwards
      // through the choice sequence rather than forwards,
      // because later bits tend to depend on earlier bits
      // so it's easier to make changes near the end and
      // deleting bits at the end may allow us to make
      // changes earlier on that we we'd have missed.
      //
      // Note that we do not restart the loop at the end
      // when we find a successful shrink. This is because
      // things we've already tried are less likely to work.
      //
      // If this guess is wrong, that's OK, this isn't a
      // correctness problem, because if we made a successful
      // reduction then we are not at a fixed point and
      // will restart the loop at the end the next time
      // round. In some cases this can result in performance
      // issues, but the end result should still be fine.
      //
      // First try deleting each choice we made in chunks.
      // We try longer chunks because this allows us to
      // delete whole composite elements: e.g. deleting an
      // element from a generated list requires us to delete
      // both the choice of whether to include it and also
      // the element itself, which may involve more than one
      // choice. Some things will take more than 8 choices
      // in the sequence. That's too bad, we may not be
      // able to delete those. In Hypothesis proper we
      // record the boundaries corresponding to ``any``
      // calls so that we can try deleting those, but
      // that's pretty high overhead and also a bunch of
      // slightly annoying code that it's not worth porting.
      //
      // We could instead do a quadratic amount of work
      // to try all boundaries, but in general we don't
      // want to do that because even a shrunk test case
      // can involve a relatively large number of choices.
      let k = 8;
      while (k > 0) {
        let i = this.result.length - k - 1;
        while (i >= 0) {
          if (i >= this.result.length) {
            // Can happen if we successfully lowered
            // the value at i - 1
            i -= 1;
            continue;
          }
          const attempt = BigintArray.from(this.result!
            .view()
            .slice(0, i)
          ).concat(this.result!.view().slice(i + k));

          assert(attempt.length < this.result.length,
            "attempt should be smaller than current result");

          if (!consider(attempt)) {
            // This fixes a common problem that occurs
            // when you have dependencies on some
            // length parameter. e.g. draw a number
            // between 0 and 10 and then draw that
            // many elements. This can't delete
            // everything that occurs that way, but
            // it can delete some things and often
            // will get us unstuck when nothing else
            // does.
            if (i > 0 && attempt[i - 1]! > 0) {
              attempt[i - 1]! -= 1n;
              if (consider(attempt)) {
                i += 1;
              }
            }
            i -= 1;
          }
        }
        k -= 1;
      }

      // Attempts tor replace some indices in the current result with new values.
      // Useful for some purely lexicographic reductions that we are about to perform.
      const replace = (values: Readonly<Map<number, bigint>>): boolean => {
        assert(!!this.result, "no result to replace values in");
        let attempt = BigintArray.from(this.result.view());
        for (const [i, v] of values.entries()) {
          // The size of this.result can change during shrinking. If that happens,
          // stop attempting to make use of these replacements because some other 
          // shrink pass is better to run now.
          if (i >= attempt.length) {
            return false;
          }
          attempt.put(i, BigInt(v));
        }

        return consider(attempt);
      };

      // Now we try replacing blocks of choices with zeroes. Note that unlike the above 
      // we skip k = 1 because we handle that in the next step. Often (but not always)
      // a block of all zeroes is the shortlex smallest value that a region can be.
      k = 8;
      while (k > 1) {
        let i = this.result.length - k;

        const m = new Map<number, bigint>();
        for (let j = i; j < i + k; j++) {
          m.set(j, 0n);
        }

        if (replace(m)) {
          // If we succeeded then all of [i, i + k] is zero so we adjust i so that the 
          // next region does not overlap with this at all.
          i -= k;
        } else {
          // Other we might still be able to zero some other these values but not the last 
          // one, so we just go back one. 
          i -= 1;
        }
      }

      // Now try replacing each choice with a smaller value by doing a binary search. This 
      // will replace n with 0 or n - 1 if possible, but will also more efficiently replace it
      // with a smaller number than doing multiple subtractions would.
      let i = this.result.length - 1;
      while (i >= 0) {
        binSearchDown(0n, this.result[i]!,
          (v: bigint) => replace(new Map<number, bigint>([[i, v]])));
        i -= 1;
      }

      // NB from here on this is just showing off cool shrinker tricks 
      // TODO@iangregson :: add the cool shrinker tricks 
    }
  }
}

// Returns n in [lo, hi] such that f(n) is true, where it is assumed and will not be checked 
// that f(hi) is true.
//
// Will return ``lo`` if ``f(lo)`` is true, otherwise the only guarantee that is made is that 
// ``f(n - 1)`` is false and ``f(n)`` is true. In particular this does *not* guarantee to 
// find the smallest value, only a locally minimal one.
function binSearchDown(lo: bigint, hi: bigint, f: (v: bigint) => boolean): bigint {
  if (f(lo)) {
    return lo;
  }

  while (lo + 1n < hi) {
    let mid = lo + (hi - lo) / 2n;
    if (f(mid)) {
      hi = mid
    } else {
      lo = mid
    }
  }

  return hi;
}

// Check is valid uint64 
function isValidUint64(n: bigint): boolean {
  return n >= 0n && n < (1n << 64n);
}
