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

test.todo("function cache", () => {
  const tf = (testCase: TestCase) => {
    if (testCase.choice(1000n) > 200n) {
      testCase.markStatus(Status.INTERESTING);
    }
    if (testCase.choice(1n) === 0n) {
      testCase.reject();
    }
  }

  const rng = new Rng();
  const maxExamples = 100;
  const state = new TestingState(rng, tf, maxExamples);
  const cached = new CachedTestFunction(state.testFn);

  expect(cached.call([1n, 1n])).toBe(Status.VALID);
  expect(cached.call([1n])).toBe(Status.OVERRUN);
  expect(cached.call([1000n])).toBe(Status.INTERESTING);
  expect(cached.call([1000n])).toBe(Status.INTERESTING);
  expect(cached.call([1000n, 1n])).toBe(Status.INTERESTING);

  expect(state.calls).toBe(2);

  /// Fails with max call stack exceeded
});

describe.todo("test max examples not exceeded", () => {
  for (let maxExamples = 0; maxExamples < 100; maxExamples++) {
    const name = `test max examples not exceeded (maxExamples=${maxExamples})`;

    test(name, () => {
      let calls = 0;
      
      expect(() => {
        runTest(name, {
          database: new MemoryDb(),
          random: new Rng(),
          maxExamples: maxExamples,
          quiet: false
        })((testCase) => {
          const m = 10000n;
          const n = testCase.choice(m);
          calls += 1;
          testCase.target(Number(n) * (Number(m) - Number(n)))
        });
      });

      expect(calls).toEqual(maxExamples);
    });
  }

  // Fails - calls not incremented as expected
});

describe.todo("test finds a local maximum", () => { });

describe.todo("test can target a score upwards to interesting", () => { });

describe.todo("test can target a score upwards without failing", () => { });

describe.todo("test targeting when most do not benefit", () => { });

describe.todo("test can target a score downwards", () => { });

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

test.todo("test can draw mixture", () => {
  runTest("test can draw mixture", {})((tc) => {
    const m = tc.any(mixOf(integers(-5n, 0n), integers(2n, 5n)));
    expect((-5n <= m && m <= 5n) && m !== 1n).toBe(true);
  });

  // Fails with eval error
});

test.todo("test target and reduce", () => { });

test.todo("test impossible weighted", () => { });

test.todo("test guaranteed weighted", () => { });

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


