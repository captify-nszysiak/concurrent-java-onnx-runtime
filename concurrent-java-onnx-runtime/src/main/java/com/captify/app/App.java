package com.captify.app;

import ai.onnxruntime.*;

import java.nio.LongBuffer;
import java.util.Arrays;
import java.util.Collections;


public class App 
{
    public static void main( String[] args ) throws OrtException {
        var env = OrtEnvironment.getEnvironment();
        var sessionOpts = new OrtSession.SessionOptions();
        // the size of the CPU thread pool used for executing multiple request concurrently
        sessionOpts.setInterOpNumThreads(10);

        var session = env.createSession("concurrent-java-onnx-runtime/src/main/resources/models/model.onnx");

        long[] inputData = {1, 2, 3};
        long[] inputShape = {1, 3};

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputData), inputShape);

        String inputName = session.getInputNames().stream().toList().get(0);
        var output = session.run(Collections.singletonMap(inputName, inputTensor));

        var outputData = output.get(0);

        System.out.println(Arrays.deepToString((float[][]) outputData.getValue()));

        inputTensor.close();
        session.close();
        env.close();
    }
}
