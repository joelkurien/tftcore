#ifndef LINEAR_H
#define LINEAR_H

#include "autograd.h"
#include "tensor_fac.h"
#include "RootLayer.h"

class Linear : public RootLayer {
    std::shared_ptr<TensorX> weights;
    std::shared_ptr<TensorX> bias;
    bool initialize;
    size_t in_feat;
    size_t out_feat;
    public:
        Linear() = default;

        Linear(size_t in_features, size_t out_features, bool init = false) 
            : RootLayer(), in_feat(in_features), 
              out_feat(out_features), initialize(init) {
            bias = tensor::deep_create({1, out_features}, true);
            weights = tensor::deep_create({in_features, out_feat});
            if(initialize){
               weights->get_data().xavier_ud(-0.5, 0.5);
            }
        }    

        virtual std::shared_ptr<TensorX> forward(std::shared_ptr<TensorX> input) override {
            try{
                size_t last_dim = input->get_data().ndim()-1;
                size_t in_features = input->get_data().shape()[last_dim];
                std::shared_ptr<TensorX> linear_calculation = add(matmul(input, weights), bias);
                return linear_calculation;
            }
            catch(const std::exception& err) {
                std::cout<<"Error in Linear Layer: "<<err.what()<<std::endl;
            }
            return input; 
        }

        virtual std::vector<std::shared_ptr<TensorX>> parameters() override {
            return {weights, bias};
        }

        ~Linear() override = default;

};

#endif
