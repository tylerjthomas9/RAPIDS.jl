"""
    module RAPIDS

A Julia interface to the RAPIDS AI ecosystem
"""
module RAPIDS

using PythonCall

const cudf = PythonCall.pynew()
const cuml = PythonCall.pynew()
const cugraph = PythonCall.pynew()
const cusignal = PythonCall.pynew()
const cuspatial = PythonCall.pynew()
const cuxfilter = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(cudf, pyimport("cudf"))
    PythonCall.pycopy!(cuml, pyimport("cuml"))
    PythonCall.pycopy!(cugraph, pyimport("cugraph"))
    PythonCall.pycopy!(cusignal, pyimport("cusignal"))
    PythonCall.pycopy!(cuspatial, pyimport("cuspatial"))
    PythonCall.pycopy!(cuxfilter, pyimport("cuxfilter"))
end


end