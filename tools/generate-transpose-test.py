#!/usr/bin/env python
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import codecs
import math
import os
import re
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xngen
import xnncommon

parser = argparse.ArgumentParser(
    description="Matrix transpose microkernel test generator")
parser.add_argument(
    "-s",
    "--spec",
    metavar="FILE",
    required=True,
    help="Specification (YAML) file")
parser.add_argument(
    "-o",
    "--output",
    metavar="FILE",
    required=True,
    help="Output (C++ source) file")
parser.set_defaults(defines=list())


def split_ukernel_name(name):
  match = re.match(r"^(xnn_(x32)_transpose_ukernel__(\d+x\d+)_(.+))_(\d+)x(\d+)$", name)
  if match is None:
    raise ValueError("Unexpected microkernel name: " + name)
  kernel = match.group(1)
  height = int(match.group(5))
  width = int(match.group(6))

  arch, isa = xnncommon.parse_target_name(target_name=match.group(4))
  return height, width, kernel, arch, isa


CVT_TEST_TEMPLATE = """\
TEST(${TEST_NAME}, block_${HEIGHT}_${WIDTH}) {
  $if ISA_CHECK:
    ${ISA_CHECK};
  TransposeMicrokernelTester()
    .height(${HEIGHT})
    .width(${WIDTH})
    .h_block_size(${HEIGHT})
    .w_block_size(${WIDTH})
    .offset(1)
    .Test(${KERNEL});
}
"""


def generate_test_cases(ukernel, init_fn, height, width, kernel, isa):
  """Generates all tests cases for a Vector Convert Operation micro-kernel.

  Args:
    ukernel: C name of the micro-kernel function.
    init_fn: C name of the function to initialize microkernel parameters.
    isa: instruction set required to run the micro-kernel. Generated unit test
      will skip execution if the host processor doesn't support this ISA.

  Returns:
    Code for the test case.
  """
  _, test_name = ukernel.split("_", 1)
  test_args = [ukernel]
  if init_fn:
    test_args.append(init_fn)
  return xngen.preprocess(
      CVT_TEST_TEMPLATE, {
          "TEST_NAME": test_name.upper().replace("UKERNEL_", ""),
          "TEST_ARGS": test_args,
          "HEIGHT": height,
          "WIDTH": width,
          "KERNEL": kernel,
          "ISA_CHECK": xnncommon.generate_isa_check_macro(isa),
      })


def main(args):
  options = parser.parse_args(args)

  with codecs.open(options.spec, "r", encoding="utf-8") as spec_file:
    spec_yaml = yaml.safe_load(spec_file)
    if not isinstance(spec_yaml, list):
      raise ValueError("expected a list of micro-kernels in the spec")

    tests = """\
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: {specification}
//   Generator: {generator}


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"
""".format(
    specification=options.spec, generator=sys.argv[0])

    for ukernel_spec in spec_yaml:
      name = ukernel_spec["name"]
      init_fn = ukernel_spec.get("init")
      height, width, kernel, arch, isa = split_ukernel_name(name)

      # specification can override architecture
      arch = ukernel_spec.get("arch", arch)

      test_case = generate_test_cases(name, init_fn, height, width, kernel, isa)
      tests += "\n\n" + xnncommon.postprocess_test_case(test_case, arch, isa)

    txt_changed = True
    if os.path.exists(options.output):
      with codecs.open(options.output, "r", encoding="utf-8") as output_file:
        txt_changed = output_file.read() != tests

    if txt_changed:
      with codecs.open(options.output, "w", encoding="utf-8") as output_file:
        output_file.write(tests)


if __name__ == "__main__":
  main(sys.argv[1:])
