/*
Programmer: Chris Tralie
Purpose: A wrapper for Python around fast Dynamic Time warping code
*/

#include <Python.h>
#include <numpy/arrayobject.h>
#include "Alignments.h"

/* Docstrings */
static char module_docstring[] =
    "This module provides an implementation of Dynamic Time Warping";
static char dtw_docstring[] =
    "Perform Dynamic Time Warping on a cross-similarity matrix";
static char dtwc_docstring[] = "Perform Dynamic Time Warping on a cross-similarity while enforcing a constraint";
static char smwat_docstring[] = "Perform Smith Waterman on a cross-similarity matrix";
static char smwatc_docstring[] = "Perform Smith Waterman on a cross-similarity matrix while enforcing a constraint";

/* Available functions */
static PyObject* SequenceAlignment_dtw(PyObject* self, PyObject* args);
static PyObject* SequenceAlignment_dtwc(PyObject* self, PyObject* args);
static PyObject* SequenceAlignment_smwat(PyObject* self, PyObject* args);
static PyObject* SequenceAlignment_smwatc(PyObject* self, PyObject* args);

/* Module specification */
static PyMethodDef module_methods[] = {
    {"DTW", SequenceAlignment_dtw, METH_VARARGS, dtw_docstring},
    {"DTWConstrained", SequenceAlignment_dtwc, METH_VARARGS, dtwc_docstring},
    {"SMWat", SequenceAlignment_smwat, METH_VARARGS, smwat_docstring},
    {"SMWatConstrained", SequenceAlignment_smwatc, METH_VARARGS, smwatc_docstring},
    {NULL, NULL, 0, NULL}
};

/* Initialize the module */
PyMODINIT_FUNC init_SequenceAlignment(void)
{
    PyObject *m = Py_InitModule3("_SequenceAlignment", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
}

static PyObject *SequenceAlignment_dtw(PyObject *self, PyObject *args)
{
    PyObject *S_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "O", &S_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(S_array, 0);
    int N = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform DTW */
    double score = DTW(S, M, N, 0, 0, M-1, N-1);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}

static PyObject *SequenceAlignment_dtwc(PyObject *self, PyObject *args)
{
    PyObject *S_obj;
    int ci, cj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oii", &S_obj, &ci, &cj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(S_array, 0);
    int N = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform DTW */
    double score = DTWConstrained(S, M, N, ci, cj);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}

static PyObject *SequenceAlignment_smwat(PyObject *self, PyObject *args)
{
    PyObject *S_obj;
    double hvPenalty;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Od", &S_obj, &hvPenalty))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(S_array, 0);
    int N = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform DTW */
    double score = SMWat(S, M, N, 0, 0, M-1, N-1, hvPenalty);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}

static PyObject *SequenceAlignment_smwatc(PyObject *self, PyObject *args)
{
    PyObject *S_obj;
    int ci, cj;
    double hvPenalty;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "Oiid", &S_obj, &ci, &cj, &hvPenalty))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *S_array = PyArray_FROM_OTF(S_obj, NPY_DOUBLE, NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (S_array == NULL) {
        Py_XDECREF(S_array);
        return NULL;
    }

    int M = (int)PyArray_DIM(S_array, 0);
    int N = (int)PyArray_DIM(S_array, 1);

    /* Get pointers to the data as C-types. */
    double *S    = (double*)PyArray_DATA(S_array);

    /* Perform DTW */
    double score = SMWatConstrained(S, M, N, ci, cj, hvPenalty);

    /* Clean up. */
    Py_DECREF(S_array);

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", score);
    return ret;
}
