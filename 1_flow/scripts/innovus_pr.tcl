
cd ./apr 
set dc_id "b14"
set run_name $dc_id
append run_name "_0.7_medium_0.9" 
file mkdir $run_name
file copy -force ../zhiyao.tcl ../apr/$run_name 
file copy -force $dc_id/output2.v ../output2.v 
file copy -force $dc_id/output2.sdc ../output2.sdc 
source ../test.globals 
init_design 
setDesignMode -process 45 
set_interactive_constraint_modes [all_constraint_modes -active] 
reset_wire_load_model 
reset_ideal_network [get_nets *] 
cd ./$run_name 

set design "b14"
clearGlobalNets 
globalNetConnect VDD -type pgpin -pin VDD -inst * -module {} 
globalNetConnect VDD -type tiehi -pin VDD -inst * -module {} 
globalNetConnect VSS -type pgpin -pin VSS -inst * -module {} 
globalNetConnect VSS -type tielo -pin VSS -inst * -module {} 
getIoFlowFlag 
setIoFlowFlag 0 
floorPlan -site FreePDK45_38x28_10R_NP_162NW_34O -r 1 0.7 30 30 30 30 
uiSetTool select 
getIoFlowFlag 
set sprCreateIeRingNets {} 
set sprCreateIeRingLayers {} 
set sprCreateIeRingWidth 1.0 
set sprCreateIeRingSpacing 1.0 
set sprCreateIeRingOffset 1.0 
set sprCreateIeRingThreshold 1.0 
set sprCreateIeRingJogDistance 1.0 
addRing -skip_via_on_wire_shape Noshape -use_wire_group_bits 2 -use_interleaving_wire_group 1 -skip_via_on_pin Standardcell -stacked_via_top_layer metal10 -use_wire_group 1 -type core_rings -jog_distance 0.095 -threshold 0.095 -nets {VDD VSS} -follow core -stacked_via_bottom_layer metal1 -layer {bottom metal5 top metal5 right metal4 left metal4} -width 5 -spacing 2 -offset 0.095 
set sprCreateIeStripeNets {} 
set sprCreateIeStripeLayers {} 
set sprCreateIeStripeWidth 2.5 
set sprCreateIeStripeSpacing 2.0 
set sprCreateIeStripeThreshold 1.0 
addStripe -skip_via_on_wire_shape Noshape -block_ring_top_layer_limit metal5 -max_same_layer_jog_length 1.6 -padcore_ring_bottom_layer_limit metal3 -number_of_sets 5 -split_vias 1 -skip_via_on_pin Standardcell -same_sized_stack_vias 1 -stacked_via_top_layer metal10 -padcore_ring_top_layer_limit metal5 -spacing 2 -xleft_offset 20 -switch_layer_over_obs 1 -xright_offset 20 -merge_stripes_value 0.095 -layer metal4 -block_ring_bottom_layer_limit metal3 -width 2.5 -nets {VDD VSS} -stacked_via_bottom_layer metal1 -break_stripes_at_block_rings 1 
addStripe -skip_via_on_wire_shape Noshape -block_ring_top_layer_limit metal6 -max_same_layer_jog_length 1.6 -padcore_ring_bottom_layer_limit metal4 -number_of_sets 5 -ybottom_offset 20 -split_vias 1 -skip_via_on_pin Standardcell -same_sized_stack_vias 1 -stacked_via_top_layer metal10 -padcore_ring_top_layer_limit metal6 -spacing 2 -switch_layer_over_obs 1 -merge_stripes_value 0.095 -direction horizontal -layer metal5 -block_ring_bottom_layer_limit metal4 -ytop_offset 20 -width 2.5 -nets {VDD VSS} -stacked_via_bottom_layer metal1 -break_stripes_at_block_rings 1 
sroute -connect { corePin } -layerChangeRange { metal1 metal10 } -blockPinTarget { nearestTarget } -corePinTarget { firstAfterRowEnd } -allowJogging 1 -crossoverViaLayerRange { metal1 metal10 } -nets { VDD VSS } -allowLayerChange 1 -targetViaLayerRange { metal1 metal10 } 
setEndCapMode -reset 
setEndCapMode -boundary_tap false 

#################
setPlaceMode -reset 
setPlaceMode -congEffort medium -timingDriven 1 -modulePlan 1 -clkGateAware 1 -powerDriven 1 -ignoreScan 1 -reorderScan 1 -ignoreSpare 0 -placeIOPins 1 -moduleAwareSpare 0 -maxDensity 0.9 -preserveRouting 0 -rmAffectedRouting 0 -checkRoute 0 -swapEEQ 0 
setPlaceMode -fp false 
placeDesign -noPrePlaceOpt 
source dumpNets.tcl

#################
report_power -outfile place_power.txt 
report_area -out_file place_area.txt 
summaryReport -outfile place_summary.txt 
exit 

