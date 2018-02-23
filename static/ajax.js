$(document).ready(function(){
    $("#ask_button").click(function(){
        if($("#paragraph").val()=="" || $("#paragraph").val()=="Please enter text."){
            $('#paragraph').val("Please enter text.");
            $('#answer').val("");
        } else if ($("#question").val()=="" || $("#question").val()=="Please enter question."){
            $('#question').val("Please enter question.");
            $('#answer').val("");
        } else {
            $.ajax({
                type: "POST",
                url: "http://127.0.0.1:5000/predict",
                dataType: "text",
                data:
                {
                    paragraph: $('#paragraph').val(),
                    question: $('#question').val()
                },
                success: function(data) {
                    $("#answer").val(data);
                },
                error: function(response, textStatus) {
                    $("#answer").val(textStatus);
                }
            });
        }
    });
});

//https://stackoverflow.com/questions/17909646/counting-and-limiting-words-in-a-textarea
$(document).ready(function() {
  $("#paragraph").on('keydown', function() {
    var words = this.value.match(/\S+/g).length;
    $('#display_count').text(words);
    if (words > 200) {
      // Split the string on first 200 words and rejoin on spaces
      var trimmed = $(this).val().split(/\s+/, 200).join(" ");
      // Add a space at the end to make sure more typing creates new words
      $(this).val(trimmed + " ");
    }
    else {
      $('#display_count').text(words);
    }
  });
});

//https://stackoverflow.com/questions/68485/how-to-show-loading-spinner-in-jquery
jQuery.ajaxSetup({
  beforeSend: function() {
     $("#answer").val("Loading, please wait...");
  },
  complete: function(){},
  success: function() {}
});

$(document).ready(function() {
    $("#answer").val("");
});