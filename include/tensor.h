#ifndef TENSOR_H
#define TENSOR_H

#include<vector>
#include<optional>
#include<cmath>
#include<numeric>
#include<omp.h>
#include<cblas.h>
#include<sstream>
#include "nditerator.h"

class Tensor {
    private:
        std::vector<double> valvec;
        std::vector<size_t> shapes;
        std::vector<size_t> strides;
        double* basePtr;
        size_t dim;

        //generalized elementwise arithematic operations
        template<typename TensorOp>
        Tensor tensorOp(const Tensor& t, TensorOp op){
            Tensor a = (t.ndim() > dim) ? singleton_rule(t) : *this;
            Tensor b = (t.ndim() < dim) ? singleton_rule(t) : t;

            std::vector<size_t> ans_shape = broadcast_shape(t);
            size_t total_size = std::accumulate(ans_shape.begin(), ans_shape.end(), size_t{1}, std::multiplies<size_t>());
            std::vector<double> new_valvec(total_size);
            std::vector<std::vector<size_t>> indices;
            for(const auto& idx: NDRange(ans_shape)){
                indices.push_back(idx);
            }

            #pragma omp parallel for if(total_size> 10000)
            for(size_t vidx=0; vidx<indices.size(); vidx++){
                const auto& idx = indices[vidx];

                std::vector<size_t> idx_a(a.ndim()), idx_b(b.ndim());

                size_t o_a = ans_shape.size() - a.ndim();
                size_t o_b = ans_shape.size() - b.ndim();

                for(size_t i=0; i<a.ndim(); i++){
                    idx_a[i] = (a.shapes[i] == 1) ? 0 : idx[i+o_a];
                }

                for(size_t i=0; i<b.ndim(); i++){
                    idx_b[i] = (b.shapes[i] == 1) ? 0 : idx[i+o_b];
                }

                double left = a.at(idx_a), right = b.at(idx_b);
                new_valvec[vidx] = op(left, right);
            }

            return Tensor(new_valvec, ans_shape);
        }

        //generalized scalar matrix operations;
        template<typename TensorOp>
        Tensor scalarOp(double val, TensorOp op){
            size_t total_size = std::accumulate(shapes.begin(), shapes.end(), size_t{1}, std::multiplies<size_t>());
            std::vector<double> new_valvec(total_size);
            std::vector<std::vector<size_t>> indices;
            for(const auto& idx: NDRange(shapes)){
                indices.push_back(idx);
            }

            #pragma omp parallel for if(total_size>10000)
            for(size_t vidx=0; vidx<total_size; vidx++){
                new_valvec[vidx] = op(at(indices[vidx]), val);
            }

            return Tensor(new_valvec, shapes); 
        }

        //reduced axis shapes and vectors for reduction operations such as sum, mean and max along axes
        std::tuple<std::vector<size_t>, std::vector<size_t>> axis_reduction(const size_t axis);
        std::vector<size_t> index(const size_t idx);    

    protected:
        size_t jumpTo(std::vector<size_t> pos) const;
        const std::vector<size_t> computeStrides(std::vector<size_t> shps) const;

    public:
        Tensor() {dim = 0;};
        Tensor(std::vector<size_t> shape_list);
        Tensor(std::vector<double> vec, std::vector<size_t> shape_list);
        Tensor(double* ptr, std::vector<size_t> shape_list, std::vector<size_t> strides);
        Tensor(std::vector<double> vec, std::vector<size_t> shape_list, std::vector<size_t> _strides);
        Tensor(double* ptr, std::vector<size_t> shape_list);
        //copy constructor
        Tensor(const Tensor& other);
        Tensor& operator= (const Tensor& other);

        double* data();
        const double* data() const;

        const std::vector<double> as_vector_const() const;
        std::vector<double>& as_vector(); 
        size_t ndim() const;
        std::vector<size_t> get_strides() const;
        const std::vector<size_t>& shape() const;
        size_t size() const;
        bool empty() const;
        const bool is_contiguous() const;
        Tensor contiguous() const;
        Tensor view(std::vector<size_t> new_shape);

//region broadcasting rules
        bool shape_check(std::vector<size_t> t_shp);
        std::vector<size_t> broadcast_shape(const Tensor& t);
        Tensor singleton_rule(const Tensor& t);
        Tensor unsqueeze(size_t axis);
        Tensor squeeze(const std::optional<size_t> axis = std::nullopt);
        Tensor expand(std::vector<size_t> target);
        Tensor mask_filled(const Tensor& mask, double replace);

//region access and modification
        double at(std::vector<size_t> pos) const;
        void put(std::vector<size_t> pos, double val);
//endregion access and modification

//region data-viewing
        // referenced slicing -> the slice is still pointing to the same location as the og tensor
        Tensor slice(std::vector<size_t> start, std::vector<size_t> shape, const std::optional<std::vector<size_t>>& _strides = std::nullopt); //works
        Tensor slice(size_t start, size_t end, std::vector<size_t> shape_list);
        Tensor reshape(std::vector<size_t> new_shape); //works
        Tensor permute(const std::optional<std::vector<size_t>>& rotaxis = std::nullopt); //works
        Tensor transpose(std::optional<size_t> a1 = std::nullopt, std::optional<size_t> a2 = std::nullopt); //works
        std::vector<Tensor>split_uneven(const std::vector<size_t>& split_len, const size_t axis); //works
        std::vector<Tensor> chunk(const size_t num_heads, const size_t axis); //works
//endregion data-viewing

//region element-wise operations
        Tensor operator+ (const Tensor& t); //works
        Tensor operator+ (const double val); //works
        Tensor operator- (const Tensor& t); //works
        Tensor operator- (const double val); //works
        Tensor operator* (const Tensor& t) ; //works
        Tensor operator* (const double val); //works
        Tensor operator/ (const double val); //works
        Tensor operator/ (const Tensor& t); //works
        Tensor operator+= (const Tensor& t); //works
        Tensor operator+= (const double t); //works
        Tensor operator-= (const Tensor& t); //works
        Tensor operator-= (const double t); //works
        Tensor operator== (const Tensor& t);
        Tensor operator> (const Tensor& t);
        Tensor operator< (const Tensor& t);
        Tensor operator!= (const Tensor& t);
//endregion element-wise operations

//region reductions
        Tensor sum(const size_t axis); //works
        Tensor mean(const size_t axis); //works
        Tensor maximum(const size_t axis); //works
        Tensor minimum(const size_t axis); //works
//endregion reductions
//
        //element-wise functions
        Tensor sqrt();
        Tensor log();
        Tensor exp();
        Tensor pow(const double n);


        //functional operations

        // Softmax function
        Tensor softmax(const size_t axis);
        Tensor log_softmax(const size_t axis);
        // Layer Normalization
        Tensor layer_norm(Tensor gamma, Tensor beta, const size_t axis);
        Tensor relu();
        Tensor gelu();
        Tensor sigmoid();
        Tensor tanh();
        Tensor elu(const double alpha);
        
        //Initializers
        void xavier_ud(const double fan_in, const double fan_out);
        Tensor dropout(const double p, const bool training, Tensor& mask);
        //Extra
        void make2d(std::vector<size_t>& shape_list, const size_t axis = 1);
};

Tensor replace(const Tensor& mask, const Tensor& a, const Tensor& b);
Tensor elemental_max(const Tensor& a, const Tensor& b);
Tensor concatenate(const std::vector<Tensor>& tensor_list, const size_t axis); //works
Tensor dot(Tensor x, Tensor y, const size_t axis);

Tensor ones(std::vector<size_t> shape);
void prnt(std::vector<size_t> x);
void prntd(std::vector<double> x);

template <typename T>
std::string vec_string(std::vector<T> vec){
    std::ostringstream output;
    output<<"(";
    for(size_t i=0; i<vec.size(); i++){
        output << vec[i];
        if(i < vec.size()-1) output << ", ";
    }
    output<<")";
    return output.str();
}

inline Tensor arange(std::initializer_list<size_t> list){
    std::vector<size_t> shapes(list);
    Tensor tensor(shapes);
    double i=1;
    for(std::vector<size_t> idx: NDRange(tensor.shape())){
        tensor.put(idx, i++);
    }
    return tensor;
}

#endif
