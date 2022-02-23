#foreach net [ get_object_name [get_nets  * -hierarchical] ] {
#    puts "$net"
#}
#
#

proc netSize { net } {
    return [sizeof_collection [get_cells -of_objects $net -hierarchical]]
}

proc checkNets { fd } {
    foreach_in_collection net  [get_nets  * -hierarchical]  {
        if {[netSize $net] > 15} {
            puts $fd "LargeFan"
        }
        if {[netSize $net] > 50} {
            puts $fd "VeryLargeFan"
        }
        puts $fd "fanout: [expr [netSize $net] - 1]"
        puts $fd "net name: [get_object_name $net]"

        #set dri [get_cells -of_objects [get_property $net driver_pins]]
        set dri [get_pins -quiet [get_property $net driver_pins]]
        set port [get_ports -quiet -of_objects $net]

        #set sinks [remove_from_collection [get_cells -of_objects $net -hierarchical -leaf] $dri]
        set sinks [remove_from_collection [get_pins -quiet -of_objects $net -hierarchical -leaf] $dri]

        if {[expr [sizeof_collection $dri] == 0]} {
            puts $fd "PortsIn!!!!"

            if {[expr [sizeof_collection $port] > 0] && [expr {[get_property $port direction] == "in"}]} {
                puts $fd "driver: [get_object_name $port] NangateOpenCellLibrary/inPort [get_property $port x_coordinate] [get_property $port y_coordinate]"
            } else {
                puts $fd "driver: NotFound NotFound NotFound NotFound"
            }
        } else  {
            puts $fd "driver: [get_object_name $dri] \
                              [get_object_name [get_lib_cells -of_objects [get_cells -of_objects $dri] ]] \
                              [get_property $dri x_coordinate] [get_property $dri y_coordinate]"
        }

        if {[expr [sizeof_collection $sinks] == 0] && [expr [sizeof_collection $port] > 0] && [expr {[get_property $port direction] == "out"}]} {
            puts $fd "PortsOut!!!!"
            puts $fd "sinks: [get_object_name $port]"
            puts $fd "sink libs: NangateOpenCellLibrary/inPort"
            puts $fd "X: [get_property $port x_coordinate]"
            puts $fd "Y: [get_property $port y_coordinate]"
        } elseif {[expr [sizeof_collection $sinks] > 0]} {
            puts $fd "sinks: [get_object_name $sinks ]"
            puts $fd "sink libs: [get_object_name [get_lib_cells -of_objects [get_cells -of_objects $sinks] ]]"
            puts $fd "X: [get_property $sinks x_coordinate ]"
            puts $fd "Y: [get_property $sinks y_coordinate ]"
        } else {
            puts $fd "sinks: NotFound"
            puts $fd "sink libs: NotFound"
            puts $fd "X: NotFound"
            puts $fd "Y: NotFound"
        }
        puts $fd "" 
    }
    puts $fd "\n End \n" 
}


set post_name ".txt"
set all_name $run_name$post_name

set fd [open $all_name w]
checkNets $fd
puts $fd "NumOfNets: [sizeof_collection [get_nets]]"
puts $fd "NumOfNets: [sizeof_collection [get_nets  * -hierarchical]]"
close $fd
puts "\n EndCall \n" 


