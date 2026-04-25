OPENQASM 2.0;
include "qelib1.inc";

qreg q[25];
creg rec[97];

reset q[0];
reset q[1];
reset q[2];
reset q[5];
reset q[6];
reset q[8];
reset q[9];
reset q[10];
reset q[13];
reset q[14];
reset q[15];
reset q[16];
reset q[17];
reset q[20];
reset q[21];
reset q[22];
reset q[24];
reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

reset q[3]; h q[3]; // decomposed RX
reset q[4]; h q[4]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[11]; h q[11]; // decomposed RX
reset q[12]; h q[12]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[23]; h q[23]; // decomposed RX
barrier q;

cx q[3], q[5];
cx q[4], q[6];
cx q[11], q[14];
cx q[12], q[15];
cx q[18], q[20];
cx q[19], q[21];
barrier q;

cx q[3], q[1];
cx q[4], q[2];
cx q[11], q[9];
cx q[12], q[10];
cx q[18], q[16];
cx q[19], q[17];
barrier q;

cx q[3], q[8];
cx q[4], q[5];
cx q[7], q[10];
cx q[11], q[17];
cx q[12], q[14];
cx q[18], q[22];
cx q[19], q[20];
barrier q;

cx q[3], q[0];
cx q[4], q[1];
cx q[11], q[6];
cx q[12], q[9];
cx q[18], q[13];
cx q[19], q[16];
cx q[23], q[21];
barrier q;

cx q[7], q[9];
cx q[11], q[16];
barrier q;

cx q[11], q[5];
cx q[23], q[20];
barrier q;

cx q[7], q[6];
cx q[11], q[13];
cx q[23], q[24];
barrier q;

cx q[7], q[2];
cx q[11], q[8];
cx q[23], q[22];
barrier q;

h q[3]; measure q[3] -> rec[0]; h q[3]; // decomposed MX
h q[4]; measure q[4] -> rec[1]; h q[4]; // decomposed MX
h q[7]; measure q[7] -> rec[2]; h q[7]; // decomposed MX
h q[11]; measure q[11] -> rec[3]; h q[11]; // decomposed MX
h q[12]; measure q[12] -> rec[4]; h q[12]; // decomposed MX
h q[18]; measure q[18] -> rec[5]; h q[18]; // decomposed MX
h q[19]; measure q[19] -> rec[6]; h q[19]; // decomposed MX
h q[23]; measure q[23] -> rec[7]; h q[23]; // decomposed MX
barrier q;

reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

cx q[5], q[3];
cx q[6], q[4];
cx q[14], q[11];
cx q[15], q[12];
cx q[20], q[18];
cx q[21], q[19];
barrier q;

cx q[1], q[3];
cx q[2], q[4];
cx q[9], q[11];
cx q[10], q[12];
cx q[16], q[18];
cx q[17], q[19];
barrier q;

cx q[8], q[3];
cx q[5], q[4];
cx q[10], q[7];
cx q[17], q[11];
cx q[14], q[12];
cx q[22], q[18];
cx q[20], q[19];
barrier q;

cx q[0], q[3];
cx q[1], q[4];
cx q[6], q[11];
cx q[9], q[12];
cx q[13], q[18];
cx q[16], q[19];
cx q[21], q[23];
barrier q;

cx q[9], q[7];
cx q[16], q[11];
barrier q;

cx q[5], q[11];
cx q[20], q[23];
barrier q;

cx q[6], q[7];
cx q[13], q[11];
cx q[24], q[23];
barrier q;

cx q[2], q[7];
cx q[8], q[11];
cx q[22], q[23];
barrier q;

measure q[3] -> rec[8];
measure q[4] -> rec[9];
measure q[7] -> rec[10];
measure q[11] -> rec[11];
measure q[12] -> rec[12];
measure q[18] -> rec[13];
measure q[19] -> rec[14];
measure q[23] -> rec[15];
barrier q;

barrier q;

reset q[3]; h q[3]; // decomposed RX
reset q[4]; h q[4]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[11]; h q[11]; // decomposed RX
reset q[12]; h q[12]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[23]; h q[23]; // decomposed RX
barrier q;

cx q[3], q[5];
cx q[4], q[6];
cx q[11], q[14];
cx q[12], q[15];
cx q[18], q[20];
cx q[19], q[21];
barrier q;

cx q[3], q[1];
cx q[4], q[2];
cx q[11], q[9];
cx q[12], q[10];
cx q[18], q[16];
cx q[19], q[17];
barrier q;

cx q[3], q[8];
cx q[4], q[5];
cx q[7], q[10];
cx q[11], q[17];
cx q[12], q[14];
cx q[18], q[22];
cx q[19], q[20];
barrier q;

cx q[3], q[0];
cx q[4], q[1];
cx q[11], q[6];
cx q[12], q[9];
cx q[18], q[13];
cx q[19], q[16];
cx q[23], q[21];
barrier q;

cx q[7], q[9];
cx q[11], q[16];
barrier q;

cx q[11], q[5];
cx q[23], q[20];
barrier q;

cx q[7], q[6];
cx q[11], q[13];
cx q[23], q[24];
barrier q;

cx q[7], q[2];
cx q[11], q[8];
cx q[23], q[22];
barrier q;

h q[3]; measure q[3] -> rec[16]; h q[3]; // decomposed MX
h q[4]; measure q[4] -> rec[17]; h q[4]; // decomposed MX
h q[7]; measure q[7] -> rec[18]; h q[7]; // decomposed MX
h q[11]; measure q[11] -> rec[19]; h q[11]; // decomposed MX
h q[12]; measure q[12] -> rec[20]; h q[12]; // decomposed MX
h q[18]; measure q[18] -> rec[21]; h q[18]; // decomposed MX
h q[19]; measure q[19] -> rec[22]; h q[19]; // decomposed MX
h q[23]; measure q[23] -> rec[23]; h q[23]; // decomposed MX
barrier q;

barrier q;

reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

cx q[5], q[3];
cx q[6], q[4];
cx q[14], q[11];
cx q[15], q[12];
cx q[20], q[18];
cx q[21], q[19];
barrier q;

cx q[1], q[3];
cx q[2], q[4];
cx q[9], q[11];
cx q[10], q[12];
cx q[16], q[18];
cx q[17], q[19];
barrier q;

cx q[8], q[3];
cx q[5], q[4];
cx q[10], q[7];
cx q[17], q[11];
cx q[14], q[12];
cx q[22], q[18];
cx q[20], q[19];
barrier q;

cx q[0], q[3];
cx q[1], q[4];
cx q[6], q[11];
cx q[9], q[12];
cx q[13], q[18];
cx q[16], q[19];
cx q[21], q[23];
barrier q;

cx q[9], q[7];
cx q[16], q[11];
barrier q;

cx q[5], q[11];
cx q[20], q[23];
barrier q;

cx q[6], q[7];
cx q[13], q[11];
cx q[24], q[23];
barrier q;

cx q[2], q[7];
cx q[8], q[11];
cx q[22], q[23];
barrier q;

measure q[3] -> rec[24];
measure q[4] -> rec[25];
measure q[7] -> rec[26];
measure q[11] -> rec[27];
measure q[12] -> rec[28];
measure q[18] -> rec[29];
measure q[19] -> rec[30];
measure q[23] -> rec[31];
barrier q;

barrier q;

reset q[3]; h q[3]; // decomposed RX
reset q[4]; h q[4]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[11]; h q[11]; // decomposed RX
reset q[12]; h q[12]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[23]; h q[23]; // decomposed RX
barrier q;

cx q[3], q[5];
cx q[4], q[6];
cx q[11], q[14];
cx q[12], q[15];
cx q[18], q[20];
cx q[19], q[21];
barrier q;

cx q[3], q[1];
cx q[4], q[2];
cx q[11], q[9];
cx q[12], q[10];
cx q[18], q[16];
cx q[19], q[17];
barrier q;

cx q[3], q[8];
cx q[4], q[5];
cx q[7], q[10];
cx q[11], q[17];
cx q[12], q[14];
cx q[18], q[22];
cx q[19], q[20];
barrier q;

cx q[3], q[0];
cx q[4], q[1];
cx q[11], q[6];
cx q[12], q[9];
cx q[18], q[13];
cx q[19], q[16];
cx q[23], q[21];
barrier q;

cx q[7], q[9];
cx q[11], q[16];
barrier q;

cx q[11], q[5];
cx q[23], q[20];
barrier q;

cx q[7], q[6];
cx q[11], q[13];
cx q[23], q[24];
barrier q;

cx q[7], q[2];
cx q[11], q[8];
cx q[23], q[22];
barrier q;

h q[3]; measure q[3] -> rec[32]; h q[3]; // decomposed MX
h q[4]; measure q[4] -> rec[33]; h q[4]; // decomposed MX
h q[7]; measure q[7] -> rec[34]; h q[7]; // decomposed MX
h q[11]; measure q[11] -> rec[35]; h q[11]; // decomposed MX
h q[12]; measure q[12] -> rec[36]; h q[12]; // decomposed MX
h q[18]; measure q[18] -> rec[37]; h q[18]; // decomposed MX
h q[19]; measure q[19] -> rec[38]; h q[19]; // decomposed MX
h q[23]; measure q[23] -> rec[39]; h q[23]; // decomposed MX
barrier q;

barrier q;

reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

cx q[5], q[3];
cx q[6], q[4];
cx q[14], q[11];
cx q[15], q[12];
cx q[20], q[18];
cx q[21], q[19];
barrier q;

cx q[1], q[3];
cx q[2], q[4];
cx q[9], q[11];
cx q[10], q[12];
cx q[16], q[18];
cx q[17], q[19];
barrier q;

cx q[8], q[3];
cx q[5], q[4];
cx q[10], q[7];
cx q[17], q[11];
cx q[14], q[12];
cx q[22], q[18];
cx q[20], q[19];
barrier q;

cx q[0], q[3];
cx q[1], q[4];
cx q[6], q[11];
cx q[9], q[12];
cx q[13], q[18];
cx q[16], q[19];
cx q[21], q[23];
barrier q;

cx q[9], q[7];
cx q[16], q[11];
barrier q;

cx q[5], q[11];
cx q[20], q[23];
barrier q;

cx q[6], q[7];
cx q[13], q[11];
cx q[24], q[23];
barrier q;

cx q[2], q[7];
cx q[8], q[11];
cx q[22], q[23];
barrier q;

measure q[3] -> rec[40];
measure q[4] -> rec[41];
measure q[7] -> rec[42];
measure q[11] -> rec[43];
measure q[12] -> rec[44];
measure q[18] -> rec[45];
measure q[19] -> rec[46];
measure q[23] -> rec[47];
barrier q;

barrier q;

reset q[3]; h q[3]; // decomposed RX
reset q[4]; h q[4]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[11]; h q[11]; // decomposed RX
reset q[12]; h q[12]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[23]; h q[23]; // decomposed RX
barrier q;

cx q[3], q[5];
cx q[4], q[6];
cx q[11], q[14];
cx q[12], q[15];
cx q[18], q[20];
cx q[19], q[21];
barrier q;

cx q[3], q[1];
cx q[4], q[2];
cx q[11], q[9];
cx q[12], q[10];
cx q[18], q[16];
cx q[19], q[17];
barrier q;

cx q[3], q[8];
cx q[4], q[5];
cx q[7], q[10];
cx q[11], q[17];
cx q[12], q[14];
cx q[18], q[22];
cx q[19], q[20];
barrier q;

cx q[3], q[0];
cx q[4], q[1];
cx q[11], q[6];
cx q[12], q[9];
cx q[18], q[13];
cx q[19], q[16];
cx q[23], q[21];
barrier q;

cx q[7], q[9];
cx q[11], q[16];
barrier q;

cx q[11], q[5];
cx q[23], q[20];
barrier q;

cx q[7], q[6];
cx q[11], q[13];
cx q[23], q[24];
barrier q;

cx q[7], q[2];
cx q[11], q[8];
cx q[23], q[22];
barrier q;

h q[3]; measure q[3] -> rec[48]; h q[3]; // decomposed MX
h q[4]; measure q[4] -> rec[49]; h q[4]; // decomposed MX
h q[7]; measure q[7] -> rec[50]; h q[7]; // decomposed MX
h q[11]; measure q[11] -> rec[51]; h q[11]; // decomposed MX
h q[12]; measure q[12] -> rec[52]; h q[12]; // decomposed MX
h q[18]; measure q[18] -> rec[53]; h q[18]; // decomposed MX
h q[19]; measure q[19] -> rec[54]; h q[19]; // decomposed MX
h q[23]; measure q[23] -> rec[55]; h q[23]; // decomposed MX
barrier q;

barrier q;

reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

cx q[5], q[3];
cx q[6], q[4];
cx q[14], q[11];
cx q[15], q[12];
cx q[20], q[18];
cx q[21], q[19];
barrier q;

cx q[1], q[3];
cx q[2], q[4];
cx q[9], q[11];
cx q[10], q[12];
cx q[16], q[18];
cx q[17], q[19];
barrier q;

cx q[8], q[3];
cx q[5], q[4];
cx q[10], q[7];
cx q[17], q[11];
cx q[14], q[12];
cx q[22], q[18];
cx q[20], q[19];
barrier q;

cx q[0], q[3];
cx q[1], q[4];
cx q[6], q[11];
cx q[9], q[12];
cx q[13], q[18];
cx q[16], q[19];
cx q[21], q[23];
barrier q;

cx q[9], q[7];
cx q[16], q[11];
barrier q;

cx q[5], q[11];
cx q[20], q[23];
barrier q;

cx q[6], q[7];
cx q[13], q[11];
cx q[24], q[23];
barrier q;

cx q[2], q[7];
cx q[8], q[11];
cx q[22], q[23];
barrier q;

measure q[3] -> rec[56];
measure q[4] -> rec[57];
measure q[7] -> rec[58];
measure q[11] -> rec[59];
measure q[12] -> rec[60];
measure q[18] -> rec[61];
measure q[19] -> rec[62];
measure q[23] -> rec[63];
barrier q;

barrier q;

reset q[3]; h q[3]; // decomposed RX
reset q[4]; h q[4]; // decomposed RX
reset q[7]; h q[7]; // decomposed RX
reset q[11]; h q[11]; // decomposed RX
reset q[12]; h q[12]; // decomposed RX
reset q[18]; h q[18]; // decomposed RX
reset q[19]; h q[19]; // decomposed RX
reset q[23]; h q[23]; // decomposed RX
barrier q;

cx q[3], q[5];
cx q[4], q[6];
cx q[11], q[14];
cx q[12], q[15];
cx q[18], q[20];
cx q[19], q[21];
barrier q;

cx q[3], q[1];
cx q[4], q[2];
cx q[11], q[9];
cx q[12], q[10];
cx q[18], q[16];
cx q[19], q[17];
barrier q;

cx q[3], q[8];
cx q[4], q[5];
cx q[7], q[10];
cx q[11], q[17];
cx q[12], q[14];
cx q[18], q[22];
cx q[19], q[20];
barrier q;

cx q[3], q[0];
cx q[4], q[1];
cx q[11], q[6];
cx q[12], q[9];
cx q[18], q[13];
cx q[19], q[16];
cx q[23], q[21];
barrier q;

cx q[7], q[9];
cx q[11], q[16];
barrier q;

cx q[11], q[5];
cx q[23], q[20];
barrier q;

cx q[7], q[6];
cx q[11], q[13];
cx q[23], q[24];
barrier q;

cx q[7], q[2];
cx q[11], q[8];
cx q[23], q[22];
barrier q;

h q[3]; measure q[3] -> rec[64]; h q[3]; // decomposed MX
h q[4]; measure q[4] -> rec[65]; h q[4]; // decomposed MX
h q[7]; measure q[7] -> rec[66]; h q[7]; // decomposed MX
h q[11]; measure q[11] -> rec[67]; h q[11]; // decomposed MX
h q[12]; measure q[12] -> rec[68]; h q[12]; // decomposed MX
h q[18]; measure q[18] -> rec[69]; h q[18]; // decomposed MX
h q[19]; measure q[19] -> rec[70]; h q[19]; // decomposed MX
h q[23]; measure q[23] -> rec[71]; h q[23]; // decomposed MX
barrier q;

barrier q;

reset q[3];
reset q[4];
reset q[7];
reset q[11];
reset q[12];
reset q[18];
reset q[19];
reset q[23];
barrier q;

cx q[5], q[3];
cx q[6], q[4];
cx q[14], q[11];
cx q[15], q[12];
cx q[20], q[18];
cx q[21], q[19];
barrier q;

cx q[1], q[3];
cx q[2], q[4];
cx q[9], q[11];
cx q[10], q[12];
cx q[16], q[18];
cx q[17], q[19];
barrier q;

cx q[8], q[3];
cx q[5], q[4];
cx q[10], q[7];
cx q[17], q[11];
cx q[14], q[12];
cx q[22], q[18];
cx q[20], q[19];
barrier q;

cx q[0], q[3];
cx q[1], q[4];
cx q[6], q[11];
cx q[9], q[12];
cx q[13], q[18];
cx q[16], q[19];
cx q[21], q[23];
barrier q;

cx q[9], q[7];
cx q[16], q[11];
barrier q;

cx q[5], q[11];
cx q[20], q[23];
barrier q;

cx q[6], q[7];
cx q[13], q[11];
cx q[24], q[23];
barrier q;

cx q[2], q[7];
cx q[8], q[11];
cx q[22], q[23];
barrier q;

measure q[3] -> rec[72];
measure q[4] -> rec[73];
measure q[7] -> rec[74];
measure q[11] -> rec[75];
measure q[12] -> rec[76];
measure q[18] -> rec[77];
measure q[19] -> rec[78];
measure q[23] -> rec[79];
barrier q;

barrier q;

measure q[0] -> rec[80];
measure q[1] -> rec[81];
measure q[2] -> rec[82];
measure q[5] -> rec[83];
measure q[6] -> rec[84];
measure q[8] -> rec[85];
measure q[9] -> rec[86];
measure q[10] -> rec[87];
measure q[13] -> rec[88];
measure q[14] -> rec[89];
measure q[15] -> rec[90];
measure q[16] -> rec[91];
measure q[17] -> rec[92];
measure q[20] -> rec[93];
measure q[21] -> rec[94];
measure q[22] -> rec[95];
measure q[24] -> rec[96];
