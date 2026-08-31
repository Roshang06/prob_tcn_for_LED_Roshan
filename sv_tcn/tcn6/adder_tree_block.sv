module adder_tree_block # (parameter int NUM, DATA_WIDTH) (
    input clk, reset,
    input logic signed [0:NUM-1][DATA_WIDTH-1:0] nums,
    output logic signed [31:0] sum
);
logic signed [31:0] tmp_sum;

if (NUM == 1) begin: base_case1
    always_ff @(posedge clk) begin
        if (reset) tmp_sum <= '0;
        else tmp_sum <= 32'($signed(nums[0]));
    end
end else if (NUM == 2) begin: base_case2
    always_ff @(posedge clk) begin
        if (reset) tmp_sum <= '0;
        else tmp_sum <= 32'($signed(nums[0])) + 32'($signed(nums[1]));
    end
end else begin: recursion
    localparam halfnum = NUM >> 1;
    logic signed [31:0] sum1;
    logic signed [31:0] sum2;
    adder_tree_block # (.NUM(halfnum), .DATA_WIDTH(DATA_WIDTH)) subAdder1 (
        .clk(clk), 
        .reset(reset), 
        .nums(nums[0:halfnum-1]), 
        .sum(sum1)
    );
    adder_tree_block # (.NUM(NUM-halfnum), .DATA_WIDTH(DATA_WIDTH)) subAdder2 (
        .clk(clk), 
        .reset(reset), 
        .nums(nums[halfnum:NUM-1]), 
        .sum(sum2)
    );

    always_ff @(posedge clk) begin
        if (reset) tmp_sum <= '0;
        else tmp_sum <= sum1 + sum2;
    end
end

always_ff @(posedge clk) begin
    if (reset) sum <= '0;
    else sum <= tmp_sum;
end
endmodule