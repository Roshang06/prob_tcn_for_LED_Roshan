module Qxxmultiply # (parameter int DATA_WIDTH) (
    input clk, reset,
    input logic signed [DATA_WIDTH-1:0] input_reg, 
    input logic signed [DATA_WIDTH-1:0] weight,
    output logic signed [31:0] final_product
);

localparam qBitShift = DATA_WIDTH/2;
logic signed [31:0] product;
logic signed [31:0] product_out;
logic signed [31:0] rounded;
logic signed [DATA_WIDTH-1:0] a;
logic signed [DATA_WIDTH-1:0] b;

always_ff @(posedge clk) begin
    if (reset) begin
        a <= '0;
        b <= '0;
        product <= '0;
        product_out <= '0;
        rounded <= '0;
    end else begin
        a <= input_reg;
        b <= weight;
        product <= a * b;
        product_out <= product;
        rounded <= (product_out + (1 <<< (qBitShift-1))) >>> qBitShift;
        final_product <= rounded; 
    end
end
endmodule