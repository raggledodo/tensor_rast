FROM mkaichen/ubuntu-setup:bazel
ENV APP_DIR /usr/src/tensor_rast
WORKDIR $APP_DIR
COPY . $APP_DIR
RUN apt-get install -y git
RUN bazel build --spawn_strategy=standalone //...
RUN cp -Lr bazel-bin bin
RUN tar -czvf tensor_rast.tar.gz bin

FROM raggledodo/buildtools:tensorflow
ENV APP_DIR /usr/src/tensor_rast
WORKDIR $APP_DIR
COPY --from=0 /usr/src/tensor_rast/tensor_rast.tar.gz .
COPY --from=0 /usr/src/tensor_rast/tensor.yaml .
RUN tar -xvf tensor_rast.tar.gz
CMD [ "./bin/test" ]
