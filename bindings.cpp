#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "tensor.h"
#include "autograd.h"
#include "MatrixMultiply.h"
#include "tensor_fac.h"

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

PYBIND11_MODULE(peped, tensor) {
    //Tensor module
    py::module_ t = tensor.def_submodule("tensor",  
            "Tensor module: vector operations without auto-differentiation");

    py::class_<Tensor>(t, "Tensor", py::buffer_protocol())
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
        .def("dim", &Tensor::ndim)
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
        .def("__truediv__", py::overload_cast<double>(&Tensor::operator/))
        
        .def("__iadd__", py::overload_cast<const Tensor&>(&Tensor::operator+=))
        .def("__iadd__", py::overload_cast<const double>(&Tensor::operator+=))

        .def("__isub__", py::overload_cast<const Tensor&>(&Tensor::operator-=))
        .def("__isub__", py::overload_cast<const double>(&Tensor::operator-=))
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
        .def("dropout", [](Tensor& x, double p, bool training){
            Tensor mask;
            auto result = x.dropout(p, training, mask);
            return py::make_tuple(result, mask);
        }, py::arg("p"), py::arg("training"))

        .def("__repr__", [](const Tensor& t){
            return "Tensor(shape="+vec_string(t.shape()) + ")";
        });
    t.def("matmul", &MatrixMul::matmul, py::arg("a"), py::arg("b"));
    t.def("concatenate", &concatenate, py::arg("tensors"), py::arg("axis"));
    t.def("ones", &ones, py::arg("shape"));
    t.def("dot", &dot, py::arg("x"), py::arg("y"), py::arg("axis"));
    t.def("elemental_max",
            py::overload_cast<const Tensor&, const Tensor&>(&elemental_max),
            py::arg("a"), py::arg("b"));
    t.def("replace",
            py::overload_cast<const Tensor&, const Tensor&, const Tensor&>(&replace),
            py::arg("mask"), py::arg("a"), py::arg("b"));

    //Autograd Module
    py::module_ ag = tensor.def_submodule("autograd", 
            "TensorX: vector operations with auto differentiation and backward topological map creation.");
    py::class_<TensorX, std::shared_ptr<TensorX>>(ag, "TensorX")
        .def(py::init<Tensor, bool>(),
            py::arg("data"), py::arg("requires_grad") = true)
        
        .def("data", &TensorX::get_data, py::return_value_policy::reference_internal)
        .def("grad", &TensorX::get_grad, py::return_value_policy::reference_internal)
        .def("requires_grad", &TensorX::get_required_grad)

        .def("nd_data", [](TensorX& tx) -> py::array_t<double> {
            return py::array_t<double>(create_buffer_info(tx.get_data()));
        })
        .def("nd_grad", [](TensorX& tx) -> py::array_t<double> {
            return py::array_t<double>(create_buffer_info(tx.get_grad()));
        })

        .def("backward", [](TensorX& tx, py::object grad_base){
            if(grad_base.is_none())
                tx.backward(std::nullopt);
            else
                tx.backward(grad_base.cast<Tensor>());
        }, 
        py::arg("grad") = py::none())
        .def("zero_grad", &TensorX::grad_zeros)
        .def("accumulate", &TensorX::accumulate, py::arg("grad"))
        .def("set_backward_fn", &TensorX::set_autograd_fn, py::arg("fn"))

        .def("shape", [](TensorX& tx) { return tx.get_data().shape(); })
        .def("gshape", [](TensorX& tx) { return tx.get_grad().shape(); })

        .def("size", [](TensorX& tx) { return tx.get_data().size(); })
        .def("size", [](TensorX& tx) { return tx.get_data().size(); })

        .def("dim", [](TensorX& tx) { return tx.get_grad().ndim(); }) 
        .def("gdim", [](TensorX& tx) { return tx.get_grad().ndim(); })

        .def("__repr__", [](TensorX& tx) {
            return "TensorX(shape="+vec_string(tx.get_data().shape()) + ", requires_grad=" 
                    + (tx.get_required_grad() ? "True" : "False") + ")";
        }); 

        //arithematic operations
        tensor.def("add", 
                py::overload_cast<std::shared_ptr<TensorX>, std::shared_ptr<TensorX>>(&add)
        );
        tensor.def("add",
                py::overload_cast<std::shared_ptr<TensorX>, double>(&add)
        );
        tensor.def("subtract",
                 py::overload_cast<std::shared_ptr<TensorX>, std::shared_ptr<TensorX>>(&subtract)
        );
        tensor.def("subtract",
                 py::overload_cast<std::shared_ptr<TensorX>, double>(&subtract)
        );
        tensor.def("multiply",
                 py::overload_cast<std::shared_ptr<TensorX>, std::shared_ptr<TensorX>>(&multiply)
        );
        tensor.def("multiply",
                 py::overload_cast<std::shared_ptr<TensorX>, double>(&multiply)
        );
        tensor.def("divide",
                 py::overload_cast<std::shared_ptr<TensorX>, std::shared_ptr<TensorX>>(&divide)
        );
        tensor.def("divide",
                 py::overload_cast<std::shared_ptr<TensorX>, double>(&divide)
        );

        //mathematical functions
        ag.def("sqrt", 
                static_cast<std::shared_ptr<TensorX>(*)(std::shared_ptr<TensorX>)>(&sqrt),
                py::arg("x"));
        ag.def("log", 
                static_cast<std::shared_ptr<TensorX>(*)(std::shared_ptr<TensorX>)>(&log),
                py::arg("x"));
        ag.def("exp", 
                static_cast<std::shared_ptr<TensorX>(*)(std::shared_ptr<TensorX>)>(&exp),
                py::arg("x"));
        ag.def("pow", 
                static_cast<std::shared_ptr<TensorX>(*)(std::shared_ptr<TensorX>, const double)>(&pow),
                py::arg("x"), py::arg("n"));

        ag.def("sum", &sum, py::arg("x"), py::arg("axis"));
        ag.def("mean", &mean, py::arg("x"), py::arg("axis"));
        ag.def("var", &var, py::arg("x"), py::arg("axis"));
        ag.def("maximum", 
                py::overload_cast<std::shared_ptr<TensorX>, const size_t>(&maximum), 
                py::arg("x"), 
                py::arg("axis"));
        ag.def("minimum", 
                py::overload_cast<std::shared_ptr<TensorX>, const size_t>(&minimum), 
                py::arg("x"), 
                py::arg("axis"));
        
        ag.def("relu", &relu, py::arg("x"));
        ag.def("gelu", &gelu, py::arg("x"));
        ag.def("sigmoid", &sigmoid, py::arg("x"));
        ag.def("tanh", 
                static_cast<std::shared_ptr<TensorX>(*)(std::shared_ptr<TensorX>)>(&tanh),
                py::arg("x"));
        ag.def("elu", &elu, py::arg("x"), py::arg("alpha") = 1);
        ag.def("softmax", &softmax, py::arg("x"), py::arg("axis"));
        ag.def("log_softmax", &log_softmax, py::arg("x"), py::arg("axis"));
        ag.def("glu", &glu, py::arg("x"), py::arg("axis"));
        ag.def("reGlu", &reGlu, py::arg("x"));

        ag.def("layer_norm", &layer_norm,
                py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("axis"));

        ag.def("squeeze", &squeeze, py::arg("x"), py::arg("axis") = py::none());
        ag.def("unsqueeze", &unsqueeze, py::arg("x"), py::arg("axis"));
        ag.def("expand", &expand, py::arg("x"), py::arg("target_shape"));
        ag.def("reshape", &reshape, py::arg("x"), py::arg("new_shape"));
        ag.def("transpose", &transpose, 
                py::arg("x"), 
                py::arg("a1") = py::none(), 
                py::arg("a2") = py::none());

        ag.def("permute", &permute, py::arg("x"), py::arg("axes") = py::none());

        ag.def("chunk", &chunk, py::arg("x"), py::arg("num_chunks"), py::arg("axis"));
        ag.def("concat", &concat, py::arg("tensors"), py::arg("axis"));
        ag.def("stack", [](std::vector<std::shared_ptr<TensorX>> tensors, size_t axis){
            return stack(tensors, axis);
        }, py::arg("tensors"), py::arg("axis"));

        ag.def("slice", &slice, 
                py::arg("x"), 
                py::arg("start"), 
                py::arg("shape"), 
                py::arg("strides") = py::none());

        ag.def("fill_mask", &masked_fill, 
                py::arg("x"), 
                py::arg("mask"), 
                py::arg("replace"));

        ag.def("replace", py::overload_cast<const Tensor&,
                                                std::shared_ptr<TensorX>,
                                                std::shared_ptr<TensorX>>(&replace), 
                py::arg("mask"), 
                py::arg("x"), 
                py::arg("y"));
        ag.def("max_element", py::overload_cast<std::shared_ptr<TensorX>, 
                std::shared_ptr<TensorX>>(&elemental_max), py::arg("x"), py::arg("y"));

        ag.def("dropout", [](std::shared_ptr<TensorX> x, double p, bool training){
            Tensor mask;
            auto result = dropout(x, p, training, mask);
            return py::make_tuple(result, mask);
        }, py::arg("x"), py::arg("p"), py::arg("training"));

        ag.def("pinball_loss", &pinball_loss, py::arg("y"), py::arg("y_pred"), py::arg("tau"));
        ag.def("matmul", py::overload_cast<std::shared_ptr<TensorX>, 
                std::shared_ptr<TensorX>>(&matmul), py::arg("x"), py::arg("y"));

        //simpler tensor creation for autograd
        py::module_ fac = tensor.def_submodule("ease_tensor", "Factory helpers to create autograd tensors");

        fac.def("create", &tensor::create, 
                py::arg("data"), 
                py::arg("required_grad") = true);
        
        fac.def("deep_create", 
                py::overload_cast<std::vector<size_t>, bool>(&tensor::deep_create),
                py::arg("shape"), py::arg("requires_grad") = true);
        
        fac.def("deep_create",
                py::overload_cast<std::vector<double>, std::vector<size_t>, bool>(&tensor::deep_create),
                py::arg("data"), py::arg("shape"), py::arg("requires_grad") = true);

}

