module AdderConv # (parameter int DATA_WIDTH) (
    input clk, reset,
    input logic signed [DATA_WIDTH-1:0] input_reg, 
    input logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [DATA_WIDTH:0] adder_out
);

logic signed [31:0] sub;

always_ff @(posedge clk) begin
    if (reset) begin
        sub <= '0;
        adder_out <= '0;
    end else begin
        sub <= input_reg - weight;
        adder_out <= (sub < 0) ? sub : -sub;
    end
end
endmodule

function automatic int AdderConv_latency();
    int latency = 2;
    return latency;
endfunction