import { test, expect, describe } from "bun:test";
import { 
  CachedTestFunction,
  Errors,
  integers,
  just,
  lists,
  MemoryDb,
  mixOf,
  nothing,
  Possibility,
  Rng,
  runTest,
  setBufferSize,
  Status,
  TestCase,
  TestingState,
  toNumber,
  tuples
} from ".";

describe("test finds small list", () => {
  for (let seed = 0; seed < 10; seed++) {
    const name = `test finds small list (seed=${seed})`;

    test(name, () => {
      const outputs: string[] = [];
      const originalLog = console.log;
      console.log = (...args) => {
        outputs.push(args.join(" "));
        originalLog(...args);
      };

      expect(() => {
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(seed),
          maxExamples: 1000,
          quiet: false
        })((testCase) => {
          const ls = testCase.any(lists(integers(0n, 10000n)));
          const sum = ls.reduce((acc, val) => acc + val, 0n);
          expect(sum).toBeLessThanOrEqual(1000n);
        });
      }).toThrow();

      console.log = originalLog;
      expect(outputs[0]).toContain("any(lists(integers(0, 10000))): [1001]");
    });
  }
});

describe("test finds small list even with bad list", () => {
  for (let seed = 0; seed < 10; seed++) {
    const name = `test finds small list even with bad list (seed=${seed})`;

    test(name, () => {
      const outputs: string[] = [];
      const originalLog = console.log;
      console.log = (...args) => {
        outputs.push(args.join(" "));
        originalLog(...args);
      };

      expect(() => {
        const p = new Possibility((testCase: TestCase): bigint[] => {
          const n = testCase.choice(10n);
          return Array(Number(n)).fill(0n).map(() => testCase.choice(10000n));
        }, 'bad_list');
        
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(seed),
          maxExamples: 1000,
          quiet: false
        })((testCase) => {
          const ls = testCase.any(p);
          const sum = ls.reduce((acc, val) => acc + val, 0n);
          expect(sum).toBeLessThanOrEqual(1000n);
        });
      }).toThrow();

      console.log = originalLog;
      expect(outputs[0]).toContain("any(bad_list): [1001]");
    });
  }
});

describe("reduces additive pairs", () => {
  for (let seed = 0; seed < 10; seed++) {
    const name = `reduces additive pairs (seed=${seed})`;

    test(name, () => {
      const outputs: string[] = [];
      const originalLog = console.log;
      console.log = (...args) => {
        outputs.push(args.join(" "));
        originalLog(...args);
      };

      expect(() => {
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(seed),
          maxExamples: 1000,
          quiet: false
        })((testCase) => {
          const m = testCase.choice(1000n);
          const n = testCase.choice(1000n);
          expect(m + n).toBeLessThanOrEqual(1000n);
        });
      }).toThrow();

      console.log = originalLog;
      const choiceOutputs = outputs.filter(line => line.includes("choice(1000)"));
      expect(choiceOutputs.length).toBeGreaterThanOrEqual(2);
      expect(choiceOutputs[0]).toEqual("choice(1000): 1");
      expect(choiceOutputs[1]).toEqual("choice(1000): 1000");
    });
  }
});

describe.todo("test reuses results from the database", () => { });

test("test test cases satisfy preconditions", () => {
  expect(() => {
    runTest("test test cases satisfy preconditions", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      const n = testCase.choice(10n);
      testCase.assume(n !== 0n);
      expect(n !== 0n).toBe(true);
    });
  }).not.toThrow();
});

test("test error on too strict preconditions", () => {
  expect(() => {
    runTest("test error on too strict preconditions", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      const _ = testCase.choice(10n);
      testCase.reject();
    });
  }).toThrow(Errors.Unsatisfiable);
});

test("error on unbounded test function", () => {
  const bufferSizeController = setBufferSize(10);
  expect(() => {
    runTest("error on unbounded test function", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 5,
      quiet: false
    })((testCase) => {
      while (true) {
        testCase.choice(10n);
      }
    });
  }).toThrow(Errors.Unsatisfiable);
  bufferSizeController.restore();
});

test("function cache", () => {
  const tf = (testCase: TestCase) => {
    if (testCase.choice(1000n) > 200n) {
      testCase.markStatus(Status.INTERESTING);
    }
    if (testCase.choice(1n) === 0n) {
      testCase.reject();
    }
  }

  const rng = new Rng(0);
  const maxExamples = 100;
  const state = new TestingState(rng, tf, maxExamples);
  const cached = new CachedTestFunction(state.testFn.bind(state));

  expect(state.calls).toBe(0);
  expect(cached.call([1n, 1n])).toBe(Status.VALID);
  expect(state.calls).toBe(1);
  expect(cached.call([1n])).toBe(Status.OVERRUN);
  expect(state.calls).toBe(1);
  expect(cached.call([1000n])).toBe(Status.INTERESTING);
  expect(state.calls).toBe(2);
  expect(cached.call([1000n])).toBe(Status.INTERESTING);
  expect(state.calls).toBe(2);
  expect(cached.call([1000n, 1n])).toBe(Status.INTERESTING);
  expect(state.calls).toBe(2);
});

describe("test max examples not exceeded", () => {
  for (let maxExamples = 0; maxExamples < 100; maxExamples++) {
    const name = `test max examples not exceeded (maxExamples=${maxExamples})`;
    test(name, () => {
      expect(() => {
        let calls = 0;
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(),
          maxExamples: maxExamples,
          quiet: false
        })((testCase) => {
          const m = 10000n;
          const n = testCase.choice(m);
          calls += 1;
          testCase.target(toNumber(n * (m - n)))
        });
        expect(calls).toEqual(100);
      }).toBeTruthy();
    });
  }
});

describe("test finds a local maximum", () => {
  for (let seed = 0; seed < 10; seed++) {
    const name = `test finds a local maximum (seed=${seed})`;

    test(name, () => {
      expect(() => {
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(seed),
          maxExamples: 1000,
          quiet: false
        })((testCase) => {
          const m = testCase.choice(1000n);
          const n = testCase.choice(1000n);
          const score = Number(-((m - 500n) ** 2n + (n - 500n) ** 2n));
          testCase.target(score);
          if (m === 500n && n === 500n) {
            throw new Error('local maximum 500, 500');
          }
        });
      }).toThrow('local maximum 500, 500');
    });
  }
});


test("test can target a score upwards to interesting", () => {
  const outputs: string[] = [];
  const originalLog = console.log;
  console.log = (...args) => {
    outputs.push(args.join(" "));
    originalLog(...args);
  };

  expect(() => {
    runTest("test can target a score upwards to interesting", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      const m = testCase.choice(1000n);
      const n = testCase.choice(1000n);
      const score = n + m;
      testCase.target(toNumber(score));
      expect(score).toBeLessThan(2000n);
    });
  }).toThrow();

  console.log = originalLog;
  expect(outputs[0]).toContain("choice(1000): 1000");
  expect(outputs[1]).toContain("choice(1000): 1000");
});

test("test can target a score upwards without failing", () => {
  expect(() => {
    let maxScore = 0;
    runTest("test can target a score upwards without failing", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      const m = testCase.choice(1000n);
      const n = testCase.choice(1000n);
      const score = n + m;
      testCase.target(toNumber(score));
      maxScore = Math.max(maxScore, toNumber(score));
    });
    
    expect(maxScore).toBe(2000);
  });
});

test("test targeting when most do not benefit", () => {
  const outputs: string[] = [];
  const originalLog = console.log;
  console.log = (...args) => {
    outputs.push(args.join(" "));
    originalLog(...args);
  };

  const big = 10000n;
 
  expect(() => {
    runTest("test targeting when most do not benefit", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      testCase.choice(1000n);
      testCase.choice(1000n);
      const score = testCase.choice(big);
      testCase.target(toNumber(score));
      expect(score).toBeLessThan(big);
    });
  }).toThrow();

  console.log = originalLog;
  expect(outputs[0]).toContain("choice(1000): 0");
  expect(outputs[1]).toContain("choice(1000): 0");
  expect(outputs[2]).toContain(`choice(${big}): ${big}`);
});

test("test can target a score downwards", () => {
  const outputs: string[] = [];
  const originalLog = console.log;
  console.log = (...args) => {
    outputs.push(args.join(" "));
    originalLog(...args);
  };

  expect(() => {
    runTest("test can target a score downwards", {
      database: new MemoryDb(),
      random: new Rng(),
      maxExamples: 1000,
      quiet: false
    })((testCase) => {
      const m = testCase.choice(1000n);
      const n = testCase.choice(1000n);
      const score = n + m;
      testCase.target(-toNumber(score));
      expect(score).toBeGreaterThan(0n);
    });
  }).toThrow();

  console.log = originalLog;
  expect(outputs[0]).toContain("choice(1000): 0");
  expect(outputs[1]).toContain("choice(1000): 0");
});

test("test prints a top level weighted", () => {
  let output: string[] = [];
  const originalLog = console.log;
  console.log = (...args) => {
    output.push(args.join(" "));
    originalLog(...args);
  };

  expect(() => {
    runTest("test prints a top level weighted", {
      database: new MemoryDb(),
      maxExamples: 1000
    })((testCase) => {
      expect(testCase.weighted(0.5)).toBe(true);
    });
  }).toThrow();

  console.log = originalLog;
  expect(output[0]).toContain("weighted(0.5): false");
});

test("test errors when using frozen", () => {
  const tc = TestCase.ForChoices([0n]);
  tc.status = Status.VALID;

  expect(() => {
    tc.markStatus(Status.INTERESTING);
  }).toThrow(Errors.Frozen);

  expect(() => {
    tc.choice(10n);
  }).toThrow(Errors.Frozen);

  expect(() => {
    tc.forcedChoice(10n);
  }).toThrow(Errors.Frozen);
});

test("test errors on too large choice", () => {
  const tc = TestCase.ForChoices([0n]);
  expect(() => {
    tc.choice(2n ** 64n);
  }).toThrow();
});

test("test can choose full 64 bits", () => {
  runTest("test can choose full 64 bits", {})((tc) => {
    tc.choice(2n ** 64n - 1n);
  });
});

test("test mapped possibility", () => {
  runTest("test mapped possibility", {})((tc) => {
    const n = tc.any(integers(0n, 5n).map(n => n * 2n));
    expect(n % 2n === 0n).toBe(true);
  });
});

test("test selected possibility", () => {
  runTest("test selected possibility", {})((tc) => {
    const n = tc.any(integers(0n, 5n).satisfying(n => n % 2n === 0n));
    expect(n % 2n === 0n).toBe(true);
  });
});

test("test bound possibility", () => {
  runTest("test bound possibility", {})((tc) => {
    const [m, n] = tc.any(
      integers(0n, 5n).bind(m =>
        tuples(just(m), integers(m, m + 10n))
      )
    );

    expect(m <= n && n <= m + 10n).toBe(true);
  });
});

test("test cannot witness nothing", () => {
  expect(() => {
    runTest("test cannot witness nothing", {})((tc) => {
      tc.any(nothing());
    });
  }).toThrow(Errors.Unsatisfiable);
});

test("test cannot witness empty mixOf", () => {
  expect(() => {
    runTest("test cannot witness empty mixOf", {})((tc) => {
      tc.any(mixOf());
    });
  }).toThrow(Errors.Unsatisfiable);
});

test("test can draw mixture", () => {
  runTest("test can draw mixture", {})((tc) => {
    const m = tc.any(mixOf(integers(-5n, 0n), integers(2n, 5n)));
    expect(toNumber(m)).toBeGreaterThanOrEqual(-5);
    expect(toNumber(m)).toBeLessThanOrEqual(5);
    expect(toNumber(m)).not.toBe(1);
  });
});

test("test target and reduce", () => {
  let output: string[] = [];
  const originalLog = console.log;
  console.log = (...args) => {
    output.push(args.join(" "));
    originalLog(...args);
  };

  expect(() => {
    runTest("test target and reduce", {
      database: new MemoryDb(),
    })((testCase) => {
      const m = testCase.choice(100000n);
      testCase.target(toNumber(m));
      expect(m).toBeLessThanOrEqual(99900);
    });
  }).toThrow();

  console.log = originalLog;
  expect(output[0]).toContain("choice(100000): 99901");
});

test("test impossible weighted", () => {
  expect(() => {
    runTest("test impossible weighted", {
      database: new MemoryDb(),
    })((testCase) => {
      testCase.choice(1n);
      for (const _ of Array(10)) {
        if (testCase.weighted(0.0)) {
          throw new Error('not this one');
        }
      }
      if (testCase.choice(1n)) {
        throw new Error('failure');
      }
    });
  }).toThrow('failure');
});

test("test guaranteed weighted", () => {
  expect(() => {
    runTest("test guaranteed weighted", {
      database: new MemoryDb(),
    })((testCase) => {
      if (testCase.weighted(1.0)) {
        testCase.choice(1n);
        throw new Error('failure');
      } else {
        throw new Error('not this one');
      }
    });
  }).toThrow('failure');
});

test("test size bounds on list", () => {
  runTest("test size bounds on list", { database: new MemoryDb() })((tc) => {
    const ls = tc.any(lists(integers(0n, 10n), 1, 3));
    expect(ls.length >= 1 && ls.length <= 3).toBe(true);
  });
});

test("test forced choice bounds", () => {
  expect(() => {
    runTest("test forced choice bounds", { database: new MemoryDb() })((tc) => {
      tc.forcedChoice(2n ** 64n);
    });
  }).toThrow();
});


