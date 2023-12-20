package com.captify.app;

import ai.onnxruntime.*;

import java.util.Arrays;
import java.util.Map;


public class App 
{
    public static void main( String[] args ) throws OrtException {
        var env = OrtEnvironment.getEnvironment();
        var sessionOpts = new OrtSession.SessionOptions();
        sessionOpts.setInterOpNumThreads(10);

        // replace with concurrent-java-onnx-runtime/src/main/resources/models/model.onnx
        var session = env.createSession("/Users/nszysiak/IdeaProjects/concurrent-java-onnx-runtime/model.onnx");

        String[] inputData = {"one direction edinburgh one direction edinburgh girlguiding edinburgh young adult books edinburgh",
        "edinburgh hamster accessories hamster accessories edinburgh hamster accessories edinburgh hamster",
        "accessories edinburgh"};

        var inputTensor = OnnxTensor.createTensor(env, new StringBuffer(Arrays.toString(inputData)));

        var output = session.run(Map.of("data", inputTensor));

        var outputData = output.get(0);

        System.out.println(Arrays.deepToString((float[][]) outputData.getValue()));

        inputTensor.close();
        session.close();
        env.close();
    }
}
