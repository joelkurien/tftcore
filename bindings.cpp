#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "tensor.h"
#include "autograd.h"
#include "MatrixMultiply.h"
#include "tensor_fac.h"
#include "nditerator.h"

namespace py = pybind11;

py::buffer_info create_buffer_info(Tensor& t){
    std::vector<ssize_t> t_shape(t.shape().begin(), t.shape().end());
    std::vector<ssize_t> t_bstrides;
    for(size_t stride: t.get_strides()){
        t_bstrides.push_back(static_cast<ssize_t>(stride * sizeof(double)));
    }

    return py::buffer_info(
        t.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        static_cast<ssize_t>(t.ndim()),
        t_shape,
        t_bstrides
    );
}

Tensor from_numpy(py::array_t<double, py::array::c_style> arr){
    py::buffer_info info = arr.request();
    std::vector<size_t> shape(info.shape.begin(), info.shape.end());
    std::vector<size_t> strides;
    for(size_t stride: info.strides)
        strides.push_back(static_cast<size_t>(stride/sizeof(double)));

    std::vector<double> data(
        static_cast<double*>(info.ptr),
        static_cast<double*>(info.ptr) + info.size
    );

    return Tensor(data, shape, strides);
}

PYBIND11_MODULE(peped, m) {
    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def_buffer([](Tensor& t) -> py::buffer_info {
            return create_buffer_info(t);
        })

        //Create tensor constructors to python
        .def(py::init<std::vector<size_t>>(), py::arg("shape"))
        .def(py::init<std::vector<double>, std::vector<size_t>>(), py::arg("data"), py::arg("shape"))
        .def(py::init<std::vector<double>, std::vector<size_t>, std::vector<size_t>>(),
                py::arg("data"), py::arg("shape"), py::arg("strides"))
        
        //Tensor to numpy to tensor
        .def("to_ndarray", [](Tensor& t) -> py::array_t<double> {
            return py::array_t<double>(create_buffer_info(t));
        })
        .def_static("from_numpy", &from_numpy, py::arg("array"))

        //Informative functions about the tensor
        .def("shape", &Tensor::shape)
        .def("ndim", &Tensor::ndim)
        .def("size", &Tensor::size)
        .def("empty", &Tensor::empty)
        .def("strides", &Tensor::get_strides)
        .def("is_contiguous", &Tensor::is_contiguous)

        //Indexive viewing and insertion of values
        .def("at", &Tensor::at, py::arg("indices"))
        .def("put", &Tensor::put, py::arg("indices"), py::arg("val"))
        .def("as_vector", [](Tensor& t){ return t.as_vector_const(); })

        //Viewing the data from a different perspective
        .def("to_contiguous", &Tensor::contiguous)
        .def("reshape", &Tensor::reshape, py::arg("new_shape"))
        .def("view", &Tensor::view, py::arg("new_shape"))
        .def("permute", &Tensor::permute, py::arg("axes") = py::none())
        .def("transpose", &Tensor::transpose, 
                py::arg("a1") = py::none(), py::arg("a2") = py::none())
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("axis"))
        .def("squeeze", &Tensor::squeeze, py::arg("axis") = py::none())
        .def("expand", &Tensor::expand, py::arg("target_shape"))

        //slicing
        .def("slice", 
                py::overload_cast<
                    std::vector<size_t>,
                    std::vector<size_t>,
                    const std::optional<std::vector<size_t>>&> (&Tensor::slice),
                py::arg("start"), py::arg("shape"), py::arg("strides") = py::none())
        .def("chunk", &Tensor::chunk, py::arg("num_chunks"), py::arg("axis"))
        .def("uneven_split", &Tensor::split_uneven, 
                py::arg("split_lengths"), py::arg("axis"))

        //broadcasting and validation checks
        .def("validate_shape", &Tensor::shape_check, py::arg("t_shape"))
        .def("broadcast_shape", [](Tensor& t, const Tensor& other){
            return t.broadcast_shape(other);
        }, py::arg("other"))
        .def("mask_filled", &Tensor::mask_filled, py::arg("mask"), py::arg("replace"))

        //arithematic operations
        .def("__add__", py::overload_cast<const Tensor&>(&Tensor::operator+))
        .def("__add__", py::overload_cast<double>(&Tensor::operator+))
        .def("__radd__", [](Tensor& t, double v){return t+v; })

        .def("__sub__", py::overload_cast<const Tensor&>(&Tensor::operator-))
        .def("__sub__", py::overload_cast<double>(&Tensor::operator-))

        .def("__mul__", py::overload_cast<const Tensor&>(&Tensor::operator*))
        .def("__mul__", py::overload_cast<const double>(&Tensor::operator*))
        .def("__rmul__", [](Tensor& t, double v){return t*v; })

        .def("__truediv__", py::overload_cast<const Tensor&>(&Tensor::operator/))
        .def("__truediv__", py::overload_cast<double>(&Tensor::operator-))
        
        .def("__iadd__", py::overload_cast<const Tensor&>(&Tensor::operator+=))
        .def("__iadd__", py::overload_cast<double>(&Tensor::operator+=))

        .def("__isub__", py::overload_cast<const Tensor&>(&Tensor::operator-=))
        .def("__isub__", py::overload_cast<double>(&Tensor::operator-=))
        .def("__eq__",   py::overload_cast<const Tensor&>(&Tensor::operator==))
        .def("__gt__",   py::overload_cast<const Tensor&>(&Tensor::operator>))
        .def("__lt__",   py::overload_cast<const Tensor&>(&Tensor::operator<))
        .def("__ne__",   py::overload_cast<const Tensor&>(&Tensor::operator!=))

        //mathematical functions
        .def("sum", &Tensor::sum, py::arg("axis"))
        .def("mean", &Tensor::mean, py::arg("axis"))
        .def("maximum", &Tensor::maximum, py::arg("axis"))
        .def("minimum", &Tensor::minimum, py::arg("axis"))

        .def("sqrt", &Tensor::sqrt)
        .def("log", &Tensor::log)
        .def("exp", &Tensor::exp)
        .def("pow", &Tensor::pow, py::arg("n"))

        .def("softmax", &Tensor::softmax, py::arg("axis"))
        .def("log_softmax", &Tensor::log_softmax, py::arg("axis"))
        .def("layer_norm", &Tensor::layer_norm, 
                py::arg("gamma"), py::arg("beta"), py::arg("axis"))
        .def("relu", &Tensor::relu)
        .def("gelu", &Tensor::gelu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("tanh", &Tensor::tanh)
        .def("elu", &Tensor::elu, py::arg("alpha"))

        //initializers
        .def("xavier_ud", &Tensor::xavier_ud, py::arg("fan_in"), py::arg("fan_out"))
        .def("dropout", &Tensor::dropout, 
                py::arg("p"), py::arg("training"), py::arg("mask"))

        .def("__repr__", [](const Tensor& t){
            return "Tensor(shape="+vec_string(t.shape()) + ")";
        });
    m.def("matmul", &MatrixMul::matmul, py::arg("a"), py::arg("b"));
    m.def("concatenate", &concatenate, py::arg("tensors"), py::arg("axis"));
    m.def("ones", &ones, py::arg("shape"));
    m.def("dot", &dot, py::arg("x"), py::arg("y"), py::arg("axis"));
    m.def("elemental_max",
            py::overload_cast<const Tensor&, const Tensor&>(&elemental_max),
            py::arg("a"), py::arg("b"));
    m.def("replace",
            py::overload_cast<const Tensor&, const Tensor&, const Tensor&>(&replace),
            py::arg("mask"), py::arg("a"), py::arg("b"));
}

