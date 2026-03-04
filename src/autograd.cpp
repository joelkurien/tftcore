#include "autograd.h"
#include "nditerator.h"
#include "MatrixMultiply.h"


namespace broadcasts {
    Tensor grad_reshape(Tensor gradient, const std::vector<size_t> target_shape){
        while (gradient.ndim() > target_shape.size()) {
            gradient = gradient.sum(0);
        }
        
        for(int i = target_shape.size() - 1; i >= 0; i--){
            if(gradient.shape()[i] > 1 && target_shape[i] == 1) {
                gradient = gradient.sum(i);
                gradient = gradient.unsqueeze(i);
            } 
        }
        
        return gradient.reshape(target_shape);
    }
}

Autograd::Autograd(std::function<void()> backward_fn, std::vector<std::shared_ptr<TensorX>> autograd_inputs)
    : backward_function(backward_fn), inputs(autograd_inputs) {}

TensorX::TensorX(Tensor data, bool req_grad)
            : data_(data), required_grad(req_grad) 
{
    if(required_grad){
        grad_ = Tensor(data_.shape());
    }
}
        
Tensor& TensorX::get_data()  { return data_; }
//const Tensor& TensorX::get_data() const { return data_; }
Tensor& TensorX::get_grad() { return grad_; }
//const Tensor& TensorX::get_grad() const { return grad_; }
bool TensorX::get_required_grad() const { return required_grad; }

void TensorX::set_autograd_fn(std::shared_ptr<Autograd> fn){
    autograd_function = fn;
}

void TensorX::accumulate(Tensor& new_grad){
    if(!required_grad) return;
    if(grad_.shape() != new_grad.shape()) 
        throw std::runtime_error("Incoming gradient shapes mismatch - current gradient shape: " + vec_string(grad_.shape()) + " incoming gradient shape: " + vec_string(new_grad.shape()));

    if(grad_.size() == 0)
        grad_ = new_grad;
    else{
        grad_ = grad_ + new_grad;
    }
}        

void TensorX::grad_zeros() {
    if(required_grad)
       for(size_t i=0; i<grad_.size(); i++){
            grad_.data()[i] = 0;
       } 
}

void TensorX::backward(const std::optional<Tensor>& grad) {
    if(!required_grad) throw std::runtime_error("Gradient requirement was set to false");

    //initialize our gradient
    if(grad.has_value()){
        if(grad->shape() != data_.shape()){
            throw std::runtime_error("Gradient/Data shape mismatch");
        }
        std::vector<double> grad_values(grad->size(), 1);
        grad_ = Tensor(grad_values, grad->shape());
    }
    else {
        if(data_.size() != 1){
            throw std::runtime_error("Gradient is implictly set to scalar");
        }
        grad_ = Tensor({1.0}, data_.shape());
    }

    std::vector<std::shared_ptr<TensorX>> topological_order;
    std::unordered_set<TensorX*> visited;

    topological_sort(shared_from_this(), visited, topological_order);

    std::reverse(topological_order.begin(), topological_order.end());

    for(auto& tensor: topological_order){
        if(tensor->autograd_function){
            // std::cout<<tensor->autograd_function->op_name<<std::endl;
            tensor->autograd_function->backward_function();
        }
    }
}

void TensorX::topological_sort(std::shared_ptr<TensorX> tensor, std::unordered_set<TensorX*>& visited, std::vector<std::shared_ptr<TensorX>>& topo_order) {
   if(visited.find(tensor.get()) != visited.end()) return;

    visited.insert(tensor.get());

    if(tensor->autograd_function){
        for(auto& input: tensor->autograd_function->inputs){
            topological_sort(input, visited, topo_order);
        }
    }

    topo_order.push_back(tensor);
}

std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() * y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad() * y->get_data(), x->get_data().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad() * x->get_data(), y->get_data().shape());

        x->accumulate(grad_x);
        y->accumulate(grad_y);

    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "Multiply Tensor";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> multiply(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() * y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() * y;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Multiply Value";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() + y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad(), x->get_grad().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad(), y->get_grad().shape());
        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "Add Tensor";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> add(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() + y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Add Value";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() - y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad()*1, x->get_data().shape());
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad()*-1, y->get_data().shape());

        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "Subtract Tensor";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> subtract(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() - y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z](){
        Tensor grad_x = z->get_grad() * 1;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Subtract Value";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = x->get_data() / y->get_data();
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = broadcasts::grad_reshape(z->get_grad() / y->get_data(), x->get_data().shape());

        Tensor y_squared = y->get_data() * y->get_data();
        Tensor grad_y = broadcasts::grad_reshape(z->get_grad() * x->get_data() / y_squared * -1.0, y->get_data().shape());
        
        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "Divide Tensor";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> divide(std::shared_ptr<TensorX> x, double y){
    Tensor result = x->get_data() / y;
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,y,z](){
        Tensor grad_x = z->get_grad() / y;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Divide Value";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> softmax(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().softmax(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor s = z->get_data();
        Tensor grad_z = z->get_grad();
        Tensor s_prod = (s*grad_z).sum(axis);
        Tensor s_r = grad_z - s_prod;
        Tensor grad_x = s * s_r;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Softmax";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> log_softmax(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().log_softmax(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor s = z->get_data();
        Tensor grad_z = z->get_grad();
        Tensor grad_sum = (grad_z).sum(axis);
        Tensor grad_x = grad_z - s * grad_sum;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Log Softmax";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> layer_norm(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> gamma, std::shared_ptr<TensorX> beta, const size_t axis){
    std::shared_ptr<TensorX> mean_of_x = mean(x, axis);
    if(!x->get_data().shape_check(mean_of_x->get_data().shape())){
        mean_of_x = unsqueeze(mean_of_x, axis);
    } 
    std::shared_ptr<TensorX> centered = subtract(x, mean_of_x);
    std::shared_ptr<TensorX> squared = pow(centered, 2);
    std::shared_ptr<TensorX> variance = mean(squared, axis);

    double e = 1e-6;
    std::shared_ptr<TensorX> var = add(variance, e);
    std::shared_ptr<TensorX> std = sqrt(var);
    if(!centered->get_data().shape_check(std->get_data().shape())){
        std = unsqueeze(std, axis);
    }

    std::shared_ptr<TensorX> lnorm = divide(centered, std);
    std::shared_ptr<TensorX> mul = multiply(gamma, lnorm);
    std::shared_ptr<TensorX> res = add(mul, beta);
    return res;
}

std::shared_ptr<TensorX> maximum(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().maximum(axis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, axis](){
        Tensor grad_z = z->get_grad();
        Tensor mask = z->get_data() == x->get_data();
        Tensor grad_x = grad_z * mask;
        x->accumulate(grad_x);
    };
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Axis wise Maximum";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> minimum(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().minimum(axis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, axis](){
        Tensor grad_z = z->get_grad();
        Tensor mask = z->get_data() == x->get_data();
        Tensor grad_x = grad_z * mask;
        x->accumulate(grad_x);
    };
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Axis wise Minimum";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> sum(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().sum(axis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor grad_z = z->get_grad();
        Tensor res_unq = grad_z.unsqueeze(axis);
        Tensor res = res_unq.expand(x->get_data().shape());
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Axis wise sum";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> squeeze(std::shared_ptr<TensorX> x, std::optional<size_t> axis){
    Tensor result = x->get_data().squeeze(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, axis](){
        Tensor grad_z = z->get_grad();
        size_t ax = axis.value_or(0);
        Tensor grad_x = grad_z.unsqueeze(ax);
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Squeeze";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> unsqueeze(std::shared_ptr<TensorX> x, size_t axis){
    Tensor result = x->get_data().unsqueeze(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, axis](){
        Tensor grad_z = z->get_grad();
        grad_z = grad_z.squeeze(axis);
        x->accumulate(grad_z);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Unsqueeze";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> expand(std::shared_ptr<TensorX> x, std::vector<size_t> target){
    Tensor result = x->get_data().expand(target);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z](){
        Tensor grad_z = z->get_grad();
        Tensor grad_x = broadcasts::grad_reshape(grad_z, x->get_grad().shape()); 
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Expand";
    z->set_autograd_fn(autograd);
    return z;
}


std::shared_ptr<TensorX> mean(std::shared_ptr<TensorX> x, const size_t axis){
    Tensor result = x->get_data().mean(axis);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, axis](){
        Tensor grad_z = z->get_grad().unsqueeze(axis);
        double scale = 1.0/x->get_data().shape()[axis];
        Tensor res = (grad_z * scale).expand(x->get_data().shape());
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Mean";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> var(std::shared_ptr<TensorX> x, const size_t axis){
    std::shared_ptr<TensorX> mean_of_x = mean(x, axis);
    std::shared_ptr<TensorX> centered = subtract(x, mean_of_x);
    std::shared_ptr<TensorX> squared = pow(centered, 2);
    std::shared_ptr<TensorX> variance = mean(squared, axis);
    return variance;
}

std::shared_ptr<TensorX> relu(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().relu();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x,z, result](){
        Tensor grad_z = z->get_grad();

        std::vector<double> mask(result.size(), 0);
        const double* resPtr = result.data();

        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<result.size(); i++){
            mask[i] = resPtr[i] > 0.0 ? 1.0 : 0.0;
        }

        Tensor masked(mask, result.shape());
        Tensor res = grad_z * masked;
        x->accumulate(res);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "ReLU";
    z->set_autograd_fn(autograd);
    return z;
}

static double gelu_derivative(double x){
    const double gamma = 0.044715;
    const double sqrt_over_pi = sqrt(2.0 / std::numbers::pi);
    const double cube = x*x*x;
    const double inner = sqrt_over_pi*(x+gamma*cube);
    const double sech_squared = 1-tanh(inner)*tanh(inner);
    const double pdf = 0.5*x*sech_squared;
    const double cdf = 0.5*(1.0+tanh(inner));
    return pdf+cdf;
}

std::shared_ptr<TensorX> gelu(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().gelu();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, result](){
        Tensor grad_z = z->get_grad();

        std::vector<double> grad_values(result.size(), 0);
        const double* resPtr = result.data();
        
        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<result.size(); i++){
            grad_values[i] = gelu_derivative(resPtr[i]);
        } 

        Tensor grad(grad_values, result.shape());
        Tensor res = grad_z * grad;
        x->accumulate(res);       
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "GeLU";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> elu(std::shared_ptr<TensorX> x, const double alpha){
    Tensor result = x->get_data().elu(alpha);

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, alpha] {
        Tensor grad_z = z->get_grad();

        std::vector<double> grad_vals(x->get_data().size(), 0);
        const double* resPtr = x->get_data().data();
        
        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<x->get_data().size(); i++){
            if(resPtr[i] > 0.0){
                grad_vals[i] = grad_z.data()[i];
            }
            else {
                grad_vals[i] = grad_z.data()[i] * alpha * (std::exp(resPtr[i]));
            }
        }

        Tensor grad_x(grad_vals, x->get_data().shape());
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "eLU";
    z->set_autograd_fn(autograd);
    return z;

}

std::shared_ptr<TensorX> sigmoid(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().sigmoid();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z] {
        Tensor grad_z = z->get_grad();

        std::vector<double> grad_vals(x->get_data().size(), 0);
        const double* resPtr = x->get_data().data();
        
        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<x->get_data().size(); i++){
           grad_vals[i] = grad_z.data()[i] * z->get_data().data()[i] * (1.0 - z->get_data().data()[i]); 
        }

        Tensor grad_x(grad_vals, x->get_data().shape());
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "sigmoid";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> tanh(std::shared_ptr<TensorX> x){
    Tensor result = x->get_data().tanh();

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z] {
        Tensor grad_z = z->get_grad();

        std::vector<double> grad_vals(x->get_data().size(), 0);
        const double* resPtr = x->get_data().data();
        
        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<x->get_data().size(); i++){
           grad_vals[i] = grad_z.data()[i] * (1 - z->get_data().data()[i] * z->get_data().data()[i]); 
        }

        Tensor grad_x(grad_vals, x->get_data().shape());
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "tanh";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> glu(std::shared_ptr<TensorX> x, size_t axis){
    std::vector<std::shared_ptr<TensorX>> chunks = chunk(x, 2, axis);
    std::shared_ptr<TensorX> sig = sigmoid(chunks[0]);
    return multiply(sig, chunks[1]);
}

std::shared_ptr<TensorX> reGlu(std::shared_ptr<TensorX> x, size_t axis){
    std::vector<std::shared_ptr<TensorX>> chunks = chunk(x, 2, axis);
    std::shared_ptr<TensorX> sig = relu(chunks[0]);
    return multiply(sig, chunks[1]);
}

std::shared_ptr<TensorX> matmul(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = MatrixMul::matmul(x->get_data(), y->get_data());
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, y, z] {
        Tensor grad_z = z->get_grad();
        Tensor x_grad = MatrixMul::matmul(grad_z, y->get_data().transpose());
        Tensor y_grad = MatrixMul::matmul(x->get_data().transpose(), grad_z);
       
        x_grad = broadcasts::grad_reshape(x_grad, x->get_grad().shape());
        y_grad = broadcasts::grad_reshape(y_grad, y->get_grad().shape());
        x->accumulate(x_grad);
        y->accumulate(y_grad);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x,y});
    autograd->op_name = "Matrix Multiply";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> transpose(std::shared_ptr<TensorX> x, const std::optional<size_t> a1, const std::optional<size_t> a2){
    Tensor result = x->get_data().transpose(a1, a2);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, a1, a2] {
        Tensor grad_z = z->get_grad();
        Tensor grad_T = grad_z.transpose(a1, a2);
        x->accumulate(grad_T);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix transpose";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> permute(std::shared_ptr<TensorX> x, const std::optional<std::vector<size_t>>& rotaxis){
    Tensor result = x->get_data().permute(rotaxis);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);
    
    std::vector<size_t> inv_axis(x->get_data().ndim());
    
    if(rotaxis.has_value() && rotaxis->size() == x->get_data().ndim()){
        for(size_t i=0; i<rotaxis->size(); i++){
            inv_axis[rotaxis->at(i)] = i;
        }
    }
    else {
        std::iota(inv_axis.rbegin(), inv_axis.rend(), 0);
    }
    auto backward_fn = [x, z, inv_axis] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z.permute(inv_axis);
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix permute";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> reshape(std::shared_ptr<TensorX> x, std::vector<size_t> new_shape){
    std::vector<size_t> old_shape = x->get_data().shape(); 
    Tensor result = x->get_data().reshape(new_shape); 

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, old_shape] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z.reshape(old_shape);
        x->accumulate(grad_x);
    };
    
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix reshape";
    z->set_autograd_fn(autograd);
    return z;
}

std::vector<std::shared_ptr<TensorX>> chunk(std::shared_ptr<TensorX> x, size_t num_heads, size_t axis){
    std::vector<Tensor> results = x->get_data().chunk(num_heads, axis);
    std::vector<std::shared_ptr<TensorX>> z;
    z.reserve(results.size());
    for(Tensor& result: results) {
        z.push_back(std::make_shared<TensorX>(result, true));
    }

    auto backward_fn = [x, z, axis] {
        std::vector<Tensor> grads_z;
        grads_z.reserve(z.size());
        for(std::shared_ptr<TensorX> res: z){
            grads_z.push_back(res->get_grad());
        }
        Tensor grad_x = concatenate(grads_z, axis);
        x->accumulate(grad_x);
    };
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix chunk";
    for(std::shared_ptr<TensorX> res: z){
        res->set_autograd_fn(autograd);
    }
    return z;
}


std::shared_ptr<TensorX> concat(std::vector<std::shared_ptr<TensorX>> x, const size_t axis){

    std::vector<Tensor> tensors;
    for(std::shared_ptr<TensorX> t: x) {
        tensors.push_back(t->get_data());
    }

    Tensor result = concatenate(tensors, axis);   

    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, axis] {
        Tensor grad_z = z->get_grad();
        std::vector<size_t> split_len;
        for(std::shared_ptr<TensorX> t: x){
            split_len.push_back(t->get_data().shape()[axis]);
        }
        std::vector<Tensor> splits = grad_z.split_uneven(split_len, axis);
        for(size_t i=0; i<x.size(); i++){
            x[i]->accumulate(splits[i]);
        }
    };
    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, x);
    autograd->op_name = "Matrix concat";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> stack(std::vector<std::shared_ptr<TensorX>>& x, const size_t axis){
    std::vector<std::shared_ptr<TensorX>> stack_list;
    for(std::shared_ptr<TensorX>& t: x) {
        stack_list.push_back(unsqueeze(t, axis));
    }

    return concat(stack_list, axis);   
}

std::shared_ptr<TensorX> slice(std::shared_ptr<TensorX> x, std::vector<size_t> start, std::vector<size_t> shape, const std::optional<std::vector<size_t>>& _strides){
    Tensor result = x->get_data().slice(start, shape, _strides);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, start] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x = Tensor(x->get_data().shape());
        std::vector<size_t> shape = grad_z.shape(); 
        auto indices = NDRange(shape);
        for(const auto& idx: indices){
            std::vector<size_t> org_idx = idx; 
            for(size_t i=0; i<start.size(); i++){
                org_idx[i] += start[i];
            }
            grad_x.put(org_idx, grad_z.at(idx));
        }
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix slice";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> masked_fill(std::shared_ptr<TensorX> x, const Tensor& mask, double replace){
    Tensor result = x->get_data().mask_filled(mask, replace);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);
        
    auto backward_fn = [x, z, mask] {
        Tensor grad_z = z->get_grad();
        for(size_t i=0; i<mask.size(); i++){
            if(mask.data()[i] == 0.0) grad_z.data()[i] = 0.0;
        }
        x->accumulate(grad_z);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "Matrix mask fill";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> replace( const Tensor& mask, std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = replace(mask, x->get_data(), y->get_data());
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, y, z, mask] {
        Tensor grad_z = z->get_grad();
        Tensor grad_x(x->get_data().shape());
        Tensor grad_y(y->get_data().shape());

        #pragma omp parallel for simd schedule(static)
        for(size_t i=0; i<mask.size(); i++){
            if(mask.data()[i] == 1){
                grad_x.data()[i] = grad_z.data()[i];
                grad_y.data()[i] = 0;
            } else {
                grad_x.data()[i] = 0;
                grad_y.data()[i] = grad_z.data()[i];
            }
        }

        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "Matrix replace";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> elemental_max(std::shared_ptr<TensorX> x, std::shared_ptr<TensorX> y){
    Tensor result = elemental_max(x->get_data(), y->get_data());
    std::shared_ptr<TensorX> z =std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, y, z] {
        Tensor grad_z = z->get_grad();

        Tensor grad_x = grad_z * (x->get_data() > y->get_data());
        Tensor grad_y = grad_z * (x->get_data() < y->get_data());
        x->accumulate(grad_x);
        y->accumulate(grad_y);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x, y});
    autograd->op_name = "elemental max";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> sqrt(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().sqrt();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z / (z->get_data() * 2);

       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "sqrt";
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> exp(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().exp();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z * z->get_data();
       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "exp";
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> log(std::shared_ptr<TensorX> x){
   Tensor result = x->get_data().log();
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = grad_z / x->get_data();
       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "log";
   z->set_autograd_fn(autograd);
   return z;
}

std::shared_ptr<TensorX> pow(std::shared_ptr<TensorX> x, const double n){
   Tensor result = x->get_data().pow(n);
   std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

   auto backward_fn = [x, z, n] {
       Tensor grad_z = z->get_grad();
       Tensor grad_x = (grad_z)*(x->get_data().pow(n-1) * n);

       x->accumulate(grad_x);
   };

   std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "pow";
   z->set_autograd_fn(autograd);
   return z;

}

std::shared_ptr<TensorX> dropout(std::shared_ptr<TensorX> x, const double p, const bool training, Tensor& mask){
    if(!training || p <= 0.0) return x;

    Tensor result = x->get_data().dropout(p, training, mask);
    std::shared_ptr<TensorX> z = std::make_shared<TensorX>(result, true);

    auto backward_fn = [x, z, mask](){
        Tensor grad_z = z->get_grad();
        Tensor grad_x = grad_z * mask;
        x->accumulate(grad_x);
    };

    std::shared_ptr<Autograd> autograd = std::make_shared<Autograd>(backward_fn, std::vector{x});
    autograd->op_name = "dropout";
    z->set_autograd_fn(autograd);
    return z;
}

std::shared_ptr<TensorX> pinball_loss(std::shared_ptr<TensorX> y, std::shared_ptr<TensorX> y_pred, const double tau){
    if(tau < 0 || tau > 1) throw std::invalid_argument("Tau should lie in the range from 0 - 1");
    std::shared_ptr<TensorX> success = multiply(subtract(y, y_pred), tau);
    std::shared_ptr<TensorX> failure = multiply(subtract(y, y_pred), tau-1);
    std::shared_ptr<TensorX> result = elemental_max(success, failure);
    for(size_t i=0; i<y->get_data().ndim()-1; i++)
        result = mean(result, 0);
    result = mean(result, 1);
    return result;
}


