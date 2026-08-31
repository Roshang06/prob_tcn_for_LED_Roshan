`include "tcn_layer.sv"

module tcn #(parameter TEST, DATA_WIDTH, HIDDEN_CHANNELS, KERNEL_SIZE, LAYERS, DIALATION_BASE, parameter [0:55] MODEL_TYPE) (
    input clk, reset,
    input logic signed [DATA_WIDTH-1:0] in,
    output logic signed [DATA_WIDTH-1:0] out //todo: add a control signal and timing
);
// conections connects every input and output between layers.
logic [0:LAYERS+1][HIDDEN_CHANNELS-1:0][DATA_WIDTH-1:0] connections;
assign connections[0] = in; // these are not the same size, I am assuming in will be left padded with 0s, as well as a truncation of connections[0] when assigned to the first layer input

genvar i;
generate
    for (i = 0; i <= LAYERS; i++) begin: LayerInstantiation
        if (i == 0) begin : first_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(DIALATION_BASE ** i), .IN_CH(1), .OUT_CH(HIDDEN_CHANNELS), .LAYER_NUM(i), .KERNEL_SIZE(KERNEL_SIZE), .RESAMPLE(1), .MODEL_TYPE(MODEL_TYPE)) 
                        layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset));
        end else if (i == LAYERS) begin : readout_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(1), .IN_CH(HIDDEN_CHANNELS), .OUT_CH(1), .LAYER_NUM(i), .KERNEL_SIZE(1), .RESAMPLE(2), .MODEL_TYPE(MODEL_TYPE)) 
                        layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset));
        end else begin : regular_layer
            tcn_layer #(.TEST(TEST), .DATA_WIDTH(DATA_WIDTH), .DIALATION(DIALATION_BASE ** i), .IN_CH(HIDDEN_CHANNELS), .OUT_CH(HIDDEN_CHANNELS), .LAYER_NUM(i), .KERNEL_SIZE(KERNEL_SIZE), .RESAMPLE(0), .MODEL_TYPE(MODEL_TYPE)) 
                            layer (.in(connections[i]), .out(connections[i+1]), .clk(clk), .reset(reset));
        end
    end
endgenerate

always_ff @(posedge clk) begin
    if (reset) begin
        out <= '0;
    end else begin
        out <= connections[LAYERS+1][0];
    end
end
endmodule