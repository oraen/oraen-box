package com.oraen.box.otorch;

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

@Data
public class PropagationNode<T, U> implements Layer<T, U>  {

    Layer<T, U> reality;

    List<PropagationNode<U, ?>> forwardNodes = new ArrayList<>();

    List<PropagationNode<?, T>> backwardNodes = new ArrayList<>();

    public PropagationNode( Layer<T, U> reality) {
        this.reality = reality;
    }

    public PropagationNode() {
    }

    public void addForwardNode(PropagationNode<U, ?> node) {
        this.forwardNodes.add(node);
        node.backwardNodes.add(this);
    }


    public void addBackwardNode(PropagationNode<?, T> node) {
        this.backwardNodes.add(node);
        node.forwardNodes.add(this);
    }

    @Override
    public U[] forwardBatch(T[] data) {
        U[] re = reality.forwardBatch(data);
        for(PropagationNode<? super U, ?> node : forwardNodes) {
            node.forwardBatch(re);
        }

        return re;
    }

    @Override
    public T[] backwardBatch(U[] gradOutputBatch) {
        T[] re = reality.backwardBatch(gradOutputBatch);
        for (PropagationNode<?, ? super T> node : backwardNodes) {
            node.backwardBatch(re);
        }
        return re;
    }
}
