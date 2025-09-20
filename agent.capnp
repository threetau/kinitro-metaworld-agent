@0xbf5147e1a2a3a3b1;

interface Agent {
  ping @0 (message :Text) -> (response :Text);
  act  @1 (obs :Tensor) -> (action :Tensor);
  reset @2 ();
}

struct Tensor {
  data  @0 :Data;
  shape @1 :List(Int32);
  dtype @2 :Text;
}
