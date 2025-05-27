# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import re
import multiprocessing
from math import isclose
from typing import Union

import regex
from sympy import simplify, N
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.latex import parse_latex
from latex2sympy2 import latex2sympy


def choice_answer_clean_grader(pred: str):
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]
    pred = pred[-1]

    pred = pred.rstrip(".").rstrip("/")
    return pred


def parse_digits(num):
    num = regex.sub(",", "", str(num))
    try:
        return float(num)
    except ValueError as e:
        if num.endswith("%"):
            num = num[:-1]
            if num.endswith("\\"):
                num = num[:-1]
            try:
                return float(num) / 100
            except Exception as exc:
                return None
    return None


def is_digit(num):
    # paired with parse_digits
    return parse_digits(num) is not None


def str_to_pmatrix(input_str):
    input_str = input_str.strip()
    matrix_str = re.findall(r"\{.*,.*\}", input_str)
    pmatrix_list = []

    for m in matrix_str:
        m = m.strip("{}")
        pmatrix = "".join([r"\begin{pmatrix}", m.replace(",", "\\"), r"\end{pmatrix}"])
        pmatrix_list.append(pmatrix)

    return ", ".join(pmatrix_list)


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    is_close: bool = True,
    timeout: bool = False,
) -> bool:
    """
    Exact match of math if and only if:
    1. numerical equal: both can convert to float and are equal
    2. symbolic equal: both can convert to sympy expression and are equal
    """
    if prediction is None or reference is None:
        return False
    if str(prediction.strip().lower()) == str(reference.strip().lower()):
        return True
    if (
        reference in ["A", "B", "C", "D", "E"]
        and choice_answer_clean_grader(prediction) == reference
    ):
        return True

    try:
        if is_digit(prediction) and is_digit(reference):
            prediction = parse_digits(prediction)
            reference = parse_digits(reference)
            # number questions
            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if is_close:
                        if numeric_equal(prediction, item):
                            return True
                    else:
                        if item == prediction:
                            return True
                except Exception as ex1:
                    continue
            return False
    except Exception as e:
        pass

    if not prediction and prediction not in [0, False]:
        return False

    reference = str(reference).strip()
    prediction = str(prediction).strip()

    if "pmatrix" in prediction and "pmatrix" not in reference:
        reference = str_to_pmatrix(reference)

    pred_str, ref_str = prediction, reference
    pred_start_with_enclose1 = prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")
    pred_start_with_enclose2 = prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    if pred_start_with_enclose1 or pred_start_with_enclose2:
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str.lower() == ref_str.lower():
        return True

    if (
        regex.match(r"(\(|\[).+(\)|\])", prediction) is not None
        and regex.match(r"(\(|\[).+(\)|\])", reference) is not None
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(
                        pred_parts[i], ref_parts[i], include_percentage, is_close
                    )
                    for i in range(len(pred_parts))
                ]
            ):
                return True
    check_pred_format = (prediction.startswith("\\begin{pmatrix}")
            or prediction.startswith("\\begin{bmatrix}")) and (
            prediction.endswith("\\end{pmatrix}")
            or prediction.endswith("\\end{bmatrix}")
        )
    check_ref_format = (reference.startswith("\\begin{pmatrix}")
            or reference.startswith("\\begin{bmatrix}")
        ) and (
            reference.endswith("\\end{pmatrix}") or reference.endswith("\\end{bmatrix}")
        )
    if (
        check_pred_format
        and check_ref_format
    ):
        pred_lines = [
            line.strip()
            for line in prediction[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        ref_lines = [
            line.strip()
            for line in reference[len("\\begin{pmatrix}"): -len("\\end{pmatrix}")].split("\\\\")
            if line.strip()
        ]
        matched = True
        if len(pred_lines) == len(ref_lines):
            for pred_line, ref_line in zip(pred_lines, ref_lines):
                pred_parts = pred_line.split("&")
                ref_parts = ref_line.split("&")
                if len(pred_parts) == len(ref_parts):
                    if not all(
                        [
                            math_equal(
                                pred_parts[i],
                                ref_parts[i],
                                include_percentage,
                                is_close,
                            )
                            for i in range(len(pred_parts))
                        ]
                    ):
                        matched = False
                        break
                else:
                    matched = False
                if not matched:
                    break
        else:
            matched = False
        if matched:
            return True

    if prediction.count("=") == 1 and reference.count("=") == 1:
        pred = prediction.split("=")
        pred = f"{pred[0].strip()} - ({pred[1].strip()})"
        ref = reference.split("=")
        ref = f"{ref[0].strip()} - ({ref[1].strip()})"
        if symbolic_equal(pred, ref) or symbolic_equal(f"-({pred})", ref):
            return True
    elif (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        if math_equal(
            prediction.split("=")[1], reference, include_percentage, is_close
        ):
            return True
    elif (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        if math_equal(
            prediction, reference.split("=")[1], include_percentage, is_close
        ):
            return True

    if timeout:
        if call_with_timeout(symbolic_equal_process, prediction, reference):
            return True
    else:
        if symbolic_equal(prediction, reference):
            return True

    return False


def math_equal_process(param):
    return math_equal(param[-2], param[-1])


def numeric_equal(prediction: float, reference: float):
    return isclose(reference, prediction, rel_tol=1e-4)


def symbolic_equal(a, b):
    def _parse(s):
        for f in [parse_expr, parse_latex, latex2sympy]:
            try:
                return f(s.replace("\\\\", "\\"))
            except Exception as e:
                try:
                    return f(s)
                except Exception as e1:
                    pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception as e:
        pass

    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception as e:
        pass

    try:
        if (abs(a.lhs - a.rhs)).equals(abs(b.lhs - b.rhs)):
            return True
    except Exception as e:
        pass
    try:
        if numeric_equal(float(N(a)), float(N(b))):
            return True
    except Exception as e:
        pass

    try:
        if a.shape == b.shape:
            _a = a.applyfunc(lambda x: round(x, 3))
            _b = b.applyfunc(lambda x: round(x, 3))
            if _a.equals(_b):
                return True
    except Exception as e:
        pass
    return False


def symbolic_equal_process(a, b, output_queue):
    result = symbolic_equal(a, b)
    output_queue.put(result)


def call_with_timeout(func, *args, timeout=1, **kwargs):
    output_queue = multiprocessing.Queue()
    process_args = args + (output_queue,)
    process = multiprocessing.Process(target=func, args=process_args, kwargs=kwargs)
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return False

    return output_queue.get()


