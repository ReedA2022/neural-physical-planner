"""Explicit zero embedding into an installed rectangular classical matrix array.

The caller must materialize input padding and ordered output extraction as paid
operations. This helper describes the full physical programming/read footprint;
it does not mask inactive conductances or waive their converter/loading costs.
"""
from __future__ import annotations
import numpy as np
from .backends import fail


def _array(value,name,ranks):
    try:
        raw=np.asarray(value,dtype=object)
        if raw.ndim not in ranks or not all(1<=n<=64 for n in raw.shape):
            fail(name,'expected bounded real vector/matrix with every axis in 1..64','envelope_violation')
        if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating)) for v in raw.flat):
            fail(name,'expected real finite values without implicit bool, string or complex casts')
        result=raw.astype(float)
        if not np.isfinite(result).all():fail(name,'expected finite real values')
        return result
    except (TypeError,ValueError,OverflowError) as exc:
        if hasattr(exc,'diagnostics'):raise
        fail(name,'invalid rectangular real array: '+str(exc))


def expand_array(weights,inputs,bias,installed_rows,installed_cols):
    """Embed active rows/columns at prefix coordinates of the installed array.

    Field payloads are rejected: optical padding/fanout requires its own physical
    mechanism. A vector stays a vector; a batch keeps its row ordering.
    """
    for key,value in (('installed_rows',installed_rows),('installed_cols',installed_cols)):
        if type(value) is not int or not 1<=value<=64:
            fail('array_padding.'+key,'expected an integer in [1,64]','envelope_violation')
    if isinstance(inputs,dict):
        fail('array_padding.inputs','coherent fields cannot be padded by a digital zero embedding','domain_mismatch')
    w=_array(weights,'array_padding.weights',(2,));x=_array(inputs,'array_padding.inputs',(1,2))
    rows,cols=w.shape
    if rows>installed_rows or cols>installed_cols:
        fail('array_padding','active matrix exceeds installed rectangular dimensions','resource_infeasibility')
    if x.shape[-1]!=cols:
        fail('array_padding.inputs','input width differs from active matrix columns')
    b=None if bias is None else _array(bias,'array_padding.bias',(1,))
    if b is not None and b.shape!=(rows,):
        fail('array_padding.bias','bias width differs from active matrix rows')
    full_w=np.zeros((installed_rows,installed_cols));full_w[:rows,:cols]=w
    full_x=np.zeros((*x.shape[:-1],installed_cols));full_x[..., :cols]=x
    full_b=None
    if b is not None:
        full_b=np.zeros(installed_rows);full_b[:rows]=b
    return {'weights':full_w.tolist(),'inputs':full_x.tolist(),'bias':None if full_b is None else full_b.tolist(),
            'active_rows':rows,'active_columns':cols,'installed_rows':installed_rows,'installed_columns':installed_cols,
            'utilization':(rows*cols)/(installed_rows*installed_cols),'output_indices':list(range(rows))}
